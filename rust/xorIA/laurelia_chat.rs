/// Laurelia Chat — inferencia interactiva del port Candle.
///
/// Sin argumentos: pregunta si descarga checkpoint + tokenizer automático
/// desde HuggingFace (ScortexIA/laurelia@laurelia-llm) y abre el chat.
/// El checkpoint `.pt` de torch se lee directo con Candle (pickle), sin python.
///
/// Uso:
///   cargo run --release --bin laurelia_chat
///   cargo run --release --bin laurelia_chat -- <weights.pt|safetensors> <tokenizer.json> [flags]
///
/// Flags:
///   --prompt "texto"   genera una sola respuesta y sale
///   --max-new N        tokens a generar (default 50)
///   --temp T           temperatura (default 0.7)
///   --top-k K          (default 40)
///   --top-p P          (default 0.9)
///   --rep R            repetition penalty (default 1.2)
///   --bf16             carga los pesos en BF16 (default F32)

use candle_core::{DType, Device, Result, Tensor};
use std::io::{self, Write};
use std::time::Instant;

use hf_hub::HFClientSync;

use xlstm::blocks::laurelia::weights::Weights;
use xlstm::blocks::laurelia::{Config, LaureliaTokenizer, LLM};

const REPO_ID: &str = "ScortexIA/laurelia";
const REVISION: &str = "laurelia-llm";
const CKPT_FILE: &str = "checkpoint.pt";
const TOK_FILE: &str = "tokenizer.json";

struct GenParams {
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            max_new: 50,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.2,
        }
    }
}

fn err<T>(msg: String) -> Result<T> {
    Err(candle_core::Error::Msg(msg))
}

fn download(file: &str) -> Result<std::path::PathBuf> {
    let client = HFClientSync::new()
        .map_err(|e| candle_core::Error::Msg(format!("hf-hub api: {e}")))?;
    client
        .model("ScortexIA", "laurelia")
        .download_file()
        .filename(file)
        .revision(REVISION.to_string())
        .send()
        .map_err(|e| candle_core::Error::Msg(format!("hf download {file}: {e}")))
}

fn ask_auto() -> Result<bool> {
    print!("No pasaste modelo ni tokenizer.\n¿Descargar automático desde HF ({REPO_ID}) y abrir el chat? [si/exit] > ");
    io::stdout().flush().unwrap();
    let mut line = String::new();
    if io::stdin()
        .read_line(&mut line)
        .map_err(|e| candle_core::Error::Msg(format!("stdin: {e}")))?
        == 0
    {
        return Ok(false);
    }
    let l = line.trim().to_lowercase();
    if l == "si" || l == "s" || l == "y" || l == "yes" {
        return Ok(true);
    }
    Ok(false)
}

fn print_help() {
    println!("Comandos:");
    println!("  - Escribe un prompt para generar texto.");
    println!("  - 'len <n>': tokens a generar.");
    println!("  - 'temp <x>': temperatura.");
    println!("  - 'topk <n>' / 'topp <x>': filtros top-k / top-p.");
    println!("  - 'rep <x>': repetition penalty.");
    println!("  - 'salir' o 'exit' para terminar.");
}

fn generate(model: &LLM, tokenizer: &LaureliaTokenizer, prompt: &str, p: &GenParams) -> Result<()> {
    let ids = tokenizer
        .encode(prompt)
        .map_err(|e| candle_core::Error::Msg(format!("encode: {e}")))?;
    if ids.is_empty() {
        println!("  [prompt vacío]");
        return Ok(());
    }
    let prompt_n = ids.len();
    let input = Tensor::from_vec(ids, (1, prompt_n), model.device())?;

    let start = Instant::now();
    let out = model.generate(
        &input,
        p.max_new,
        p.temperature,
        p.top_k,
        p.top_p,
        p.repetition_penalty,
        None,
    )?;
    let elapsed = start.elapsed().as_secs_f32();

    let ids: Vec<u32> = out.reshape((out.elem_count(),))?.to_vec1()?;
    let n_gen = ids.len().saturating_sub(prompt_n);
    let text = tokenizer
        .decode(&ids)
        .map_err(|e| candle_core::Error::Msg(format!("decode: {e}")))?;
    println!("{text}\n");
    println!("  [{n_gen} tokens en {elapsed:.2}s | {:.1} tok/s]", n_gen as f32 / elapsed);
    Ok(())
}

fn chat_loop(model: &LLM, tokenizer: &LaureliaTokenizer, gen: &mut GenParams) -> Result<()> {
    println!("\n╔════════════════════════════════════════════╗");
    println!("║      LAURELIA CHAT (Candle, CPU)           ║");
    println!("╚════════════════════════════════════════════╝\n");
    print_help();
    println!();

    loop {
        print!("[max:{} temp:{}] > ", gen.max_new, gen.temperature);
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin()
            .read_line(&mut input)
            .map_err(|e| candle_core::Error::Msg(format!("stdin: {e}")))?
            == 0
        {
            break;
        }
        let input = input.trim();

        if input.eq_ignore_ascii_case("salir") || input.eq_ignore_ascii_case("exit") {
            break;
        }
        if input.is_empty() {
            continue;
        }

        let lower = input.to_lowercase();
        if let Some(rest) = lower.strip_prefix("len ") {
            if let Ok(n) = rest.trim().parse::<usize>() {
                gen.max_new = n;
                println!("  -> max_new = {n}\n");
                continue;
            }
        }
        if let Some(rest) = lower.strip_prefix("temp ") {
            if let Ok(x) = rest.trim().parse::<f32>() {
                gen.temperature = x;
                println!("  -> temperature = {x}\n");
                continue;
            }
        }
        if let Some(rest) = lower.strip_prefix("topk ") {
            if let Ok(n) = rest.trim().parse::<usize>() {
                gen.top_k = n;
                println!("  -> top_k = {n}\n");
                continue;
            }
        }
        if let Some(rest) = lower.strip_prefix("topp ") {
            if let Ok(x) = rest.trim().parse::<f32>() {
                gen.top_p = x;
                println!("  -> top_p = {x}\n");
                continue;
            }
        }
        if let Some(rest) = lower.strip_prefix("rep ") {
            if let Ok(x) = rest.trim().parse::<f32>() {
                gen.repetition_penalty = x;
                println!("  -> repetition_penalty = {x}\n");
                continue;
            }
        }
        if lower.eq("help") {
            print_help();
            println!();
            continue;
        }

        generate(model, tokenizer, input, gen)?;
        println!();
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let mut gen = GenParams::default();
    let mut bf16 = false;
    let mut one_shot = None;
    let mut model_arg = None;
    let mut tok_arg = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => {
                i += 1;
                one_shot = Some(args.get(i).cloned().unwrap_or_default());
            }
            "--max-new" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    gen.max_new = v.parse().unwrap_or(gen.max_new);
                }
            }
            "--temp" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    gen.temperature = v.parse().unwrap_or(gen.temperature);
                }
            }
            "--top-k" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    gen.top_k = v.parse().unwrap_or(gen.top_k);
                }
            }
            "--top-p" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    gen.top_p = v.parse().unwrap_or(gen.top_p);
                }
            }
            "--rep" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    gen.repetition_penalty = v.parse().unwrap_or(gen.repetition_penalty);
                }
            }
            "--bf16" => bf16 = true,
            s if s.starts_with('-') => {
                return err(format!("flag desconocido: {s}"));
            }
            pos => {
                if model_arg.is_none() {
                    model_arg = Some(pos.to_string());
                } else if tok_arg.is_none() {
                    tok_arg = Some(pos.to_string());
                } else {
                    return err(format!("argumento de más: {pos}"));
                }
            }
        }
        i += 1;
    }

    let device = Device::Cpu;
    let dtype = if bf16 { DType::BF16 } else { DType::F32 };

    let (weights_path, tok_path) = match (model_arg, tok_arg) {
        (Some(m), Some(t)) => (m, t),
        _ => {
            if !ask_auto()? {
                println!("exit");
                return Ok(());
            }
            println!("Descargando tokenizer + checkpoint de HF ({REPO_ID}@{REVISION})...");
            let ckpt = download(CKPT_FILE)?;
            let tok = download(TOK_FILE)?;
            (ckpt.to_string_lossy().to_string(), tok.to_string_lossy().to_string())
        }
    };

    println!("Cargando modelo: {weights_path} ({:?}, {:?})", dtype, device);
    let model = if weights_path.ends_with(".pt") || weights_path.ends_with(".pth") {
        Weights::load_pth(&weights_path, &Config::default(), dtype, &device)?
    } else {
        Weights::load(&weights_path, &Config::default(), dtype, &device)?
    };
    let tokenizer = LaureliaTokenizer::from_file(&tok_path)
        .map_err(|e| candle_core::Error::Msg(format!("tokenizer: {e}")))?;

    println!(
        "Config: dim={} heads={} kv_groups={} layers={} ffn={} block={} vocab={}",
        model.config.dim,
        model.config.heads,
        model.config.kv_groups,
        model.config.layers,
        model.config.ffn_dim,
        model.config.block_size,
        tokenizer.vocab_size()
    );

    if let Some(prompt) = one_shot {
        return generate(&model, &tokenizer, &prompt, &gen);
    }

    chat_loop(&model, &tokenizer, &mut gen)
}
