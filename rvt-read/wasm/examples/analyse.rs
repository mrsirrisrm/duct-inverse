//! Native run of the WASM module's pipeline: `analyse <model.rvt> [model.ifc]`.
fn main() {
    let path = std::env::args().nth(1).expect("usage: analyse <model.rvt> [model.ifc]");
    let bytes = std::fs::read(path).expect("read model");
    let ifc = std::env::args().nth(2).map(|p| std::fs::read(p).expect("read ifc"));
    let start = std::time::Instant::now();
    let report = duct_pipeline::analyse_bytes(bytes, ifc.as_deref(), &rvt::control::WalkerControl::new()).expect("analyse");
    eprintln!("analysed in {:.1} s", start.elapsed().as_secs_f64());
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
