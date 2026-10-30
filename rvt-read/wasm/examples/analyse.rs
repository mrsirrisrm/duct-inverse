//! Native run of the WASM module's pipeline: `analyse <model.rvt> [model.ifc]`,
//! or `analyse <model.ifc>` for an IFC on its own. `LW=63,125,...,8k` gives
//! the unit's sound power.
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let source = std::env::var("LW").ok().map(|lw| {
        let bands: Vec<f64> = lw.split(',').map(|v| v.trim().parse().expect("LW: numbers")).collect();
        <[f64; 8]>::try_from(bands).expect("LW: 8 bands, 63 Hz to 8 kHz")
    });
    let (revit, ifc) = match args.as_slice() {
        [ifc] if ifc.to_ascii_lowercase().ends_with(".ifc") => (None, Some(ifc)),
        [revit, rest @ ..] => (Some(revit), rest.first()),
        [] => panic!("usage: analyse <model.rvt> [model.ifc] | analyse <model.ifc>"),
    };
    let read = |p: &String| std::fs::read(p).expect("read model");
    let ifc = ifc.map(read);
    let start = std::time::Instant::now();
    let report = duct_pipeline::analyse_bytes(revit.map(read), ifc.as_deref(), source, &rvt::control::WalkerControl::new()).expect("analyse");
    eprintln!("analysed in {:.1} s", start.elapsed().as_secs_f64());
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
