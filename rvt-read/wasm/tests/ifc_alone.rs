//! An IFC on its own, end to end: a unit's supply through three ducts to two
//! diffusers, with and without the room they share, and through a plenum.

use serde_json::Value;

/// STEP for the unit, ducts and diffusers (millimetres, joined port to port,
/// one supply system), with the office (8 x 4 x 2.7 m) around both
/// diffusers if `office`, and a 1.5 x 1 x 1 m plenum between the unit and
/// the first duct, through a duct of its own, if `plenum`.
fn model(office: bool, plenum: bool) -> String {
    let mut s = String::from(
        "ISO-10303-21;\nDATA;\n#1=IFCCARTESIANPOINT((0.,0.,0.));\n#4=IFCDIRECTION((0.,0.,1.));\n\
         #5=IFCSIUNIT(*,.LENGTHUNIT.,.MILLI.,.METRE.);\n#6=IFCUNITASSIGNMENT((#5));\n#7=IFCAXIS2PLACEMENT3D(#1,$,$);\n",
    );
    let mut next = 1000;
    let mut id = || {
        next += 1;
        next
    };
    // A placement at `at` and an extruded body: the profile's corners, the depth up.
    let mut element = |s: &mut String, line: &str, at: [f64; 3], corners: [(f64, f64); 2], depth: f64| {
        let (p, a, l, pts, curve, prof, solid, rep, shape) = (id(), id(), id(), id(), id(), id(), id(), id(), id());
        let ((x0, y0), (x1, y1)) = (corners[0], corners[1]);
        s.push_str(&format!(
            "#{p}=IFCCARTESIANPOINT(({},{},{}));\n#{a}=IFCAXIS2PLACEMENT3D(#{p},$,$);\n#{l}=IFCLOCALPLACEMENT($,#{a});\n\
             #{pts}=IFCCARTESIANPOINTLIST2D((({x0},{y0}),({x1},{y0}),({x1},{y1}),({x0},{y1})));\n\
             #{curve}=IFCINDEXEDPOLYCURVE(#{pts},$,$);\n#{prof}=IFCARBITRARYCLOSEDPROFILEDEF(.AREA.,$,#{curve});\n\
             #{solid}=IFCEXTRUDEDAREASOLID(#{prof},#7,#4,{depth});\n#{rep}=IFCSHAPEREPRESENTATION($,'Body','SweptSolid',(#{solid}));\n\
             #{shape}=IFCPRODUCTDEFINITIONSHAPE($,$,(#{rep}));\n",
            at[0], at[1], at[2]
        ));
        s.push_str(&line.replace("PLACE", &format!("#{l}")).replace("SHAPE", &format!("#{shape}")));
    };
    let duct = [(-200.0, -150.0), (200.0, 150.0)];
    let diffuser = [(-300.0, -300.0), (300.0, 300.0)];
    element(&mut s, "#300=IFCUNITARYEQUIPMENT('u',$,'AHU:AHU:300',$,$,PLACE,SHAPE,'300',$);\n", [0.0, 0.0, 4000.0], [(-500.0, -500.0), (500.0, 500.0)], 1000.0);
    for (n, x) in [(310, 0.0), (320, 2000.0), (330, 6000.0)] {
        element(&mut s, &format!("#{n}=IFCDUCTSEGMENT('d{n}',$,'Rectangular Duct:Supply:{n}',$,$,PLACE,SHAPE,'{n}',$);\n"), [x, 2000.0, 3000.0], duct, 1000.0);
    }
    for (n, x) in [(340, 2000.0), (350, 6000.0)] {
        element(&mut s, &format!("#{n}=IFCAIRTERMINAL('t{n}',$,'Supply Diffuser:600 x 600:{n}',$,$,PLACE,SHAPE,'{n}',$);\n"), [x, 2000.0, 2700.0], diffuser, 50.0);
    }
    if plenum {
        element(&mut s, "#303=IFCDUCTSEGMENT('d303',$,'Rectangular Duct:Supply:303',$,$,PLACE,SHAPE,'303',$);\n", [0.0, -2000.0, 3000.0], duct, 1000.0);
        element(&mut s, "#305=IFCDUCTFITTING('p305',$,'Supply Plenum:1500 x 1000:305',$,$,PLACE,SHAPE,'305',$);\n", [0.0, 0.0, 3000.0], [(-750.0, -500.0), (750.0, 500.0)], 1000.0);
    }
    if office {
        element(&mut s, "#360=IFCSPACE('o',$,'1.01',$,$,PLACE,SHAPE,'Office',.ELEMENT.,.INTERNAL.,$);\n", [0.0, 0.0, 0.0], [(0.0, 0.0), (8000.0, 4000.0)], 2700.0);
    }
    let feed: &[(u32, u32)] = if plenum { &[(300, 303), (303, 305), (305, 310)] } else { &[(300, 310)] };
    for &(a, b) in feed.iter().chain(&[(310, 320), (310, 330), (320, 340), (330, 350)]) {
        let (pa, pb, ra, rb, c) = (id(), id(), id(), id(), id());
        s.push_str(&format!(
            "#{pa}=IFCDISTRIBUTIONPORT('p{pa}',$,$,$,$,$,$,.SOURCE.);\n#{pb}=IFCDISTRIBUTIONPORT('p{pb}',$,$,$,$,$,$,.SINK.);\n\
             #{ra}=IFCRELCONNECTSPORTTOELEMENT('r{ra}',$,$,$,#{pa},#{a});\n#{rb}=IFCRELCONNECTSPORTTOELEMENT('r{rb}',$,$,$,#{pb},#{b});\n\
             #{c}=IFCRELCONNECTSPORTS('c{c}',$,$,$,#{pa},#{pb},$);\n"
        ));
    }
    // 7 ft^3/s, about 200 L/s, to each diffuser.
    s.push_str(
        "#500=IFCPROPERTYSINGLEVALUE('Flow',$,IFCREAL(7.),$);\n#501=IFCPROPERTYSET('f',$,'Mechanical',$,(#500));\n\
         #502=IFCRELDEFINESBYPROPERTIES('g',$,$,$,(#340,#350),#501);\n\
         #510=IFCDISTRIBUTIONSYSTEM('s',$,'Supply Air',$,$,$);\n\
         #511=IFCRELASSIGNSTOGROUP('a',$,$,$,(#300,#303,#305,#310,#320,#330,#340,#350),$,#510);\nENDSEC;",
    );
    s
}

fn analyse_model(office: bool, plenum: bool) -> Value {
    let control = rvt::control::WalkerControl::new();
    duct_pipeline::analyse_bytes(None, Some(model(office, plenum).as_bytes()), None, &control).expect("analyse")
}

fn analyse(office: bool) -> Value {
    analyse_model(office, false)
}

/// The plenum is a Plenum block, not a warning. Its openings are the
/// 400 x 300 ducts, 4 m apart across it with no direct line (cos θ 0), so
/// it loses only its reverberant share: S = 8 − 0.24 m², R = (7.76 · 0.05 +
/// 0.24) / 0.95, −10 lg(0.12 / R) = 7.4 dB in every band.
#[test]
fn plenum_on_the_path() {
    let report = analyse_model(false, true);
    let blocks = report["path"]["blocks"].as_array().unwrap();
    let k = blocks.iter().position(|b| b["type"] == "Plenum").expect("a Plenum block");
    let lw = |b: &Value| b["lw_db"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>();
    let r: f64 = (7.76 * 0.05 + 0.24) / 0.95;
    let expected = -10.0 * (0.12 / r).log10();
    for (before, after) in lw(&blocks[k - 1]).iter().zip(lw(&blocks[k])) {
        assert!((before - after - expected).abs() < 0.15, "{before} -> {after}, expected {expected:.1} dB");
    }
    let warnings = report["warnings"].as_array().unwrap();
    assert!(!warnings.iter().any(|w| w.as_str().unwrap().contains("not modelled")), "{warnings:?}");
}

#[test]
fn diffusers_in_one_room_are_rated_together() {
    let (shared, alone) = (analyse(true), analyse(false));
    assert_eq!(shared["ifc"]["spaces"], 1);
    let room = &shared["path"]["room"];
    assert_eq!((room["space"].as_str(), room["terminals"].as_u64()), (Some("Office"), Some(2)), "{room}");
    assert!((room["volume_m3"].as_f64().unwrap() - 86.4).abs() < 0.1);
    // The dominant diffuser is the one overhead, 1.5 m up from the ear less
    // its body's half-depth of 25 mm.
    assert!((room["dominant"]["distance_m"].as_f64().unwrap() - 1.53).abs() < 0.01, "{room}");

    let alone_room = &alone["path"]["room"];
    assert!(alone_room["space"].is_null() && alone_room["terminals"] == 1, "{alone_room}");
    assert_eq!(alone["rooms"].as_array().unwrap().len(), 2);
    assert_eq!(shared["rooms"].as_array().unwrap().len(), 1);
    // Equal diffusers 4 m apart: the far one adds 10 lg(1 + 1.53 / 4.28),
    // 1.3 dB, everywhere.
    let lp = |v: &Value| v["path"]["receiver_spl_db"].as_array().unwrap().iter().map(|b| b.as_f64().unwrap()).collect::<Vec<_>>();
    let single = analyse_single_in_office();
    for (both, one) in lp(&shared).iter().zip(single) {
        assert!((both - one - 1.33).abs() < 0.15, "{both} vs {one}");
    }
}

/// The overhead diffuser alone at the same receiver: its block Lw through
/// Schultz at 1.53 m in the office's 86.4 m³.
fn analyse_single_in_office() -> Vec<f64> {
    use wall_duct_core::autodiff::Dual;
    use wall_duct_core::physics::room::receiver_spl;
    let report = analyse(true);
    let blocks = report["path"]["blocks"].as_array().unwrap();
    let lw: Vec<f64> = blocks.last().unwrap()["lw_db"].as_array().unwrap().iter().map(|b| b.as_f64().unwrap()).collect();
    let c = Dual::<1>::constant;
    let pwl: [Dual<1>; 8] = std::array::from_fn(|i| c(lw[i]));
    receiver_spl(&pwl, c(86.4), c(1.525)).iter().map(|b| b.value).collect()
}
