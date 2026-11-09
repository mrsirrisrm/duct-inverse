//! The web app's `evaluate`, natively: a chain's JSON on stdin, its
//! design facts as JSON on stdout.
use std::io::Read;

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("read stdin");
    let input = serde_json::from_str(&input).unwrap_or_else(|e| {
        eprintln!("bad input: {e}");
        std::process::exit(2)
    });
    match wall_duct_wasm::evaluate_input(input) {
        Ok(facts) => println!("{}", serde_json::to_string_pretty(&facts).unwrap()),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2)
        }
    }
}
