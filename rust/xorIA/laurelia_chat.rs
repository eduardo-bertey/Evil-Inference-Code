/// Laurelia Chat — inferencia interactiva del port Candle.
///
/// Uso:
///   cargo run --bin laurelia_chat -- <weights.safetensors> <tokenizer.json> [flags]
///
/// Flags:
///   --prompt "texto"   genera una sola respuesta y sale
///   --max-new N        tokens a generar (default 50)
///   --temp T           temperatura (default 0.7)
///   --top-k K          (default 40)
///   --top-p P          (default 0.9)
///   --rep R            repetition penalty (default 1.2)
///   --bf16             carga los pesos en BF16 (default F32)
///
/// Modo interactivo: `len <n>`, `temp <x>`, `topk <n>`, `topp <x>`,
/// `rep <x>`, `salir`/`exit`.

use candle_core::{DType, Device, Result};
use std::io::{self, Write};
use std::time::Instant;

use xlstm::blocks::laurelia::{Config, LaureliaTokenizer, LLM};
use xlstm::blocks::laurelia::weights::Weights;

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

fn print_help() {
    println!("Comandos:");
    println!("  - Escribe un prompt para generar texto.");
    println!("  - 'len <n>': tokens a generar.");
    println!("  - 'temp <x>': temperatura.");
    println!("  - 'topk <n>' / 'topp <x>': filtros top-k / top-p.");
    println!("  - 'rep <x>': repetition penalty.");
    println!("  - 'salir' o 'exit' para terminar.");
}

fn generate(
    model: &LLM,
    tokenizer: &LaureliaTokenizer,
    prompt: &str,
    p: &GenParams,
) -> Result<()> {
    let ids = tokenizer
        .encode(prompt)
        .map_err(|e| candle_core::Error::Msg(format!("encode: {e}")))?;
    if ids.is_empty() {
        println!("  [prompt vacío]");
        return Ok(());
    }
    let prompt_n = ids.len();
    let input = candle_core::Tensor::from_vec(ids, (1, prompt_n), model.device())?;

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
    println!("  [{n_gen} tokens en {elapsed:.2}s | {:.1} tok/s]",
             n_gen as f32 / elapsed);
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Uso: laurelia_chat <weights.safetensors> <tokenizer.json> [flags]");
        std::process::exit(1);
    }

    let weights_path = &args[1];
    let tok_path = &args[2];

    let mut gen = GenParams::default();
    let mut bf16 = false;
    let mut one_shot = None;

    let mut i = 3;
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
            other => {
                eprintln!("Flag desconocido: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let device = Device::Cpu;
    let dtype = if bf16 { DType::BF16 } else { DType::F32 };

    println!("Cargando modelo: {weights_path} ({:?}, {:?})", dtype, device);
    let model = Weights::load(weights_path, &Config::default(), dtype, &device)?;
    let tokenizer = LaureliaTokenizer::from_file(tok_path)
        .map_err(|e| candle_core::Error::Msg(format!("tokenizer: {e}")))?;

    println!("Config: dim={} heads={} kv_groups={} layers={} ffn={} block={} vocab={}",
        model.config.dim, model.config.heads, model.config.kv_groups,
        model.config.layers, model.config.ffn_dim, model.config.block_size,
        tokenizer.vocab_size());

    if let Some(prompt) = one_shot {
        return generate(&model, &tokenizer, &prompt, &gen);
    }

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

        generate(&model, &tokenizer, input, &gen)?;
        println!();
    }

    Ok(())
}
