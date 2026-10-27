//! Native run of the WASM module's pipeline: `analyse <model.rvt>`.
fn main() {
    let path = std::env::args().nth(1).expect("usage: analyse <model.rvt>");
    let bytes = std::fs::read(path).expect("read model");
    let start = std::time::Instant::now();
    let report = duct_pipeline::analyse_bytes(bytes, &rvt::control::WalkerControl::new()).expect("analyse");
    eprintln!("analysed in {:.1} s", start.elapsed().as_secs_f64());
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
