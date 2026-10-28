import { Plugin } from "@opencode/plugin"
import { execFile } from "node:child_process"
import { promisify } from "node:util"

const run = promisify(execFile)

type SearchInput = {
  query?: string
  top_k?: number
}

export default Plugin.define({
  id: "rag",
  async setup(ctx) {
    const root = ctx.location.directory
    const python = process.env.OPENCODE_RAG_PYTHON ?? process.env.PYTHON ?? "python"

    await ctx.tool.transform((editor) => {
      editor.namespace({
        name: "rag",
        description: "Busqueda semantica sobre el indice RAG local del proyecto",
      })

      editor.add({
        name: "search",
        description:
          "Busca fragmentos relevantes de codigo y docs en el indice RAG local (.rag_index). " +
          "Usalo para localizar evidencia antes de leer archivos completos. " +
          "Devuelve doc_id, offsets, score y texto.",
        input: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Consulta en lenguaje natural o terminos de codigo",
            },
            top_k: {
              type: "number",
              description: "Cantidad de fragmentos a devolver (1-20, default 6)",
              minimum: 1,
              maximum: 20,
            },
          },
          required: ["query"],
          additionalProperties: false,
        },
        execute: async (raw) => {
          const input = raw as SearchInput
          const query = String(input.query ?? "").trim()
          if (!query) return { content: "Consulta vacia." }

          const topK = Math.min(20, Math.max(1, Math.round(input.top_k ?? 6)))
          const args = [
            "-m",
            "rag",
            "search",
            query,
            "--index",
            ".rag_index",
            "--json",
            "-k",
            String(topK),
            "--max-chars",
            "1600",
          ]

          try {
            const { stdout } = await run(python, args, {
              cwd: root,
              timeout: 60_000,
              maxBuffer: 8 * 1024 * 1024,
              windowsHide: true,
              encoding: "utf8",
            })
            return { content: stdout.trim() || "Sin resultados." }
          } catch (error) {
            const detail = error instanceof Error ? error.message : String(error)
            return {
              content:
                `No se pudo consultar el indice RAG (${detail}). ` +
                `Reconstruilo con: ${python} -m rag index . --out .rag_index --quiet`,
            }
          }
        },
      })
    })
  },
})
