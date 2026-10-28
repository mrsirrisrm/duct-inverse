//! Element types from Revit family and type names, by rule. A rule answers
//! only when the name is unambiguous; anything else is `None` and goes to the
//! classifier (`eval/`). Labels are those of `eval/element_types.json`.

use crate::network::{IfcInfo, Node, NodeKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    ElbowMitred,
    ElbowMitredVanes,
    ElbowRadius,
    BranchTakeoff,
    TeeOrWye,
    Transition,
    EndCap,
    Coupling,
    VolumeDamper,
    FireSmokeDamper,
    Silencer,
    Plenum,
    VavTerminalUnit,
    DiffuserCeiling,
    GrilleOrRegister,
    ExternalCapOrLouvre,
    Fan,
    AirHandlingUnit,
    FanCoilOrHeatPump,
    NotAirside,
}

impl ElementType {
    pub fn label(self) -> &'static str {
        use ElementType::*;
        match self {
            ElbowMitred => "elbow_mitred",
            ElbowMitredVanes => "elbow_mitred_vanes",
            ElbowRadius => "elbow_radius",
            BranchTakeoff => "branch_takeoff",
            TeeOrWye => "tee_or_wye",
            Transition => "transition",
            EndCap => "end_cap",
            Coupling => "coupling",
            VolumeDamper => "volume_damper",
            FireSmokeDamper => "fire_smoke_damper",
            Silencer => "silencer",
            Plenum => "plenum",
            VavTerminalUnit => "vav_terminal_unit",
            DiffuserCeiling => "diffuser_ceiling",
            GrilleOrRegister => "grille_or_register",
            ExternalCapOrLouvre => "external_cap_or_louvre",
            Fan => "fan",
            AirHandlingUnit => "air_handling_unit",
            FanCoilOrHeatPump => "fan_coil_or_heat_pump",
            NotAirside => "not_airside",
        }
    }
}

/// Where Revit filed the element. Fittings and accessories are one category
/// here: models put dampers and silencers in either. `Other` (generic models
/// and the like) is matched against every rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    Fitting,
    Terminal,
    Equipment,
    Other,
}

impl Category {
    pub fn of(kind: &NodeKind) -> Self {
        match kind {
            NodeKind::Fitting => Category::Fitting,
            NodeKind::Terminal => Category::Terminal,
            NodeKind::Equipment => Category::Equipment,
            _ => Category::Other,
        }
    }
}

/// Lower-case words padded with spaces, split at punctuation, camelCase and
/// letter-digit boundaries: "XX_Duct_Bend_NoVanes" is " xx duct bend no vanes ".
struct Words(String);

impl Words {
    fn new(s: &str) -> Self {
        let mut spaced = String::new();
        let mut prev: Option<char> = None;
        for c in s.chars() {
            if !c.is_alphanumeric() {
                spaced.push(' ');
            } else if prev.is_some_and(|p| (p.is_lowercase() && c.is_uppercase()) || p.is_alphabetic() != c.is_alphabetic()) {
                spaced.push(' ');
            }
            if c.is_alphanumeric() {
                spaced.extend(c.to_lowercase());
            }
            prev = Some(c).filter(|c| c.is_alphanumeric());
        }
        Words(format!(" {} ", spaced.split_whitespace().collect::<Vec<_>>().join(" ")))
    }

    /// Whether any phrase is present; "a+b" needs both, anywhere.
    fn any(&self, phrases: &[&str]) -> bool {
        phrases.iter().any(|p| self.has(p))
    }

    fn has(&self, phrase: &str) -> bool {
        phrase.split('+').all(|p| self.0.contains(&format!(" {p} ")))
    }
}

use Category::{Equipment, Fitting, Terminal};
use ElementType::*;

const ALL: &[Category] = &[Fitting, Terminal, Equipment];
const INLINE: &[Category] = &[Fitting, Equipment];
const TERMINAL: &[Category] = &[Terminal];
const FITTING: &[Category] = &[Fitting];

/// First match wins, so the specific comes before the general: a "fan
/// powered terminal unit" is not a fan, a "duct silencer elbow" not an elbow,
/// a "sidewall diffuser" a grille. A `None` row abstains outright.
const RULES: &[(&[Category], &[&str], Option<ElementType>)] = &[
    // Named things that are not air-path elements, whatever else the name says.
    (ALL, &["controller", "detector", "sensor", "thermostat", "cable", "tray", "collar", "filter", "convector", "cross over", "crossover"], None),
    (INLINE, &["vav", "cav", "terminal unit", "fan powered"], Some(VavTerminalUnit)),
    (INLINE, &["fire+damper", "smoke+damper", "intumescent+damper", "fsd", "sfd", "fd"], Some(FireSmokeDamper)),
    (INLINE, &["silencer", "attenuator", "attn", "schalldämpfer", "schalldampfer"], Some(Silencer)),
    (INLINE, &["damper", "vcd", "obd"], Some(VolumeDamper)),
    (INLINE, &["plenum"], Some(Plenum)),
];

/// Units, where a name often names two: "tank exhaust fan", "duct heater,
/// fan assisted". Every row is tried; two types abstain, unless one only
/// matched inside the other's longer phrase ("fan" in "fan coil").
const UNITS: &[(&[&str], Option<ElementType>)] = &[
    (&["fcu", "fan coil", "wshp", "water source heat pump", "ducted heat pump", "ducted split", "indoor unit", "cassette"], Some(FanCoilOrHeatPump)),
    // An unqualified heat pump is as likely an outdoor air-to-water unit.
    (&["heat pump", "heater"], None),
    (
        &["ahu", "air handling", "air handler", "hru", "hrv", "erv", "mvhr", "heat recovery", "energy recovery", "rtu", "rooftop unit",
          "packaged unit", "mau", "make up air", "makeup air", "doas", "dedicated outdoor air"],
        Some(AirHandlingUnit),
    ),
    (
        &["chiller", "cooling tower", "boiler", "pump", "water heater", "tank", "heat exchanger", "condensing unit", "condenser", "calorifier"],
        Some(NotAirside),
    ),
    (&["fan", "fans", "blower"], Some(Fan)),
];

const OUTLETS: &[(&[Category], &[&str], Option<ElementType>)] = &[
    (TERMINAL, &["cap", "cowl", "louvre", "louver", "rain hood", "weather hood", "gooseneck", "roof", "weather"], Some(ExternalCapOrLouvre)),
    // AU/NZ schedule tags: SAG/RAG/EAG grilles, SAD/RAD/EAD diffusers.
    (TERMINAL, &["sidewall", "wall", "grille", "grill", "register", "reg", "transfer", "sag", "rag", "eag"], Some(GrilleOrRegister)),
    (TERMINAL, &["diffuser", "diffuseur", "diff", "swirl", "slot", "cone", "valve", "sad", "rad", "ead"], Some(DiffuserCeiling)),
    (FITTING, &["takeoff", "take off", "tap", "shoe", "spigot", "spin in", "boot"], Some(BranchTakeoff)),
    (FITTING, &["tee", "wye", "cross", "y piece", "junction", "pants", "breeches"], Some(TeeOrWye)),
    (
        FITTING,
        &["transition", "reducer", "expander", "increaser", "taper", "to round", "concentric", "eccentric"],
        Some(Transition),
    ),
];

/// The one unit type `w` names, `Some(None)` when it names rival ones, and
/// `None` when it names none.
fn unit(w: &Words) -> Option<Option<ElementType>> {
    let hits: Vec<(Option<ElementType>, Vec<&str>)> = UNITS
        .iter()
        .map(|(phrases, ty)| (*ty, phrases.iter().copied().filter(|p| w.has(p)).collect::<Vec<_>>()))
        .filter(|(_, found)| !found.is_empty())
        .collect();
    let inside = |p: &str, q: &str| p != q && format!(" {q} ").contains(&format!(" {p} "));
    let mut rivals = hits.iter().filter(|(ty, found)| {
        !found.iter().all(|p| hits.iter().any(|(other, theirs)| other != ty && theirs.iter().any(|q| inside(p, q))))
    });
    match (rivals.next(), rivals.next()) {
        (None, _) => None,
        (Some((ty, _)), None) => Some(*ty),
        _ => Some(None),
    }
}

/// Bends need the qualifier: vanes, mitred or radiused. Round elbows are
/// radiused or gored; an unqualified rectangular one is left to the classifier.
fn elbow(w: &Words) -> Option<ElementType> {
    if !w.any(&["elbow", "bend", "elb", "ell"]) {
        return None;
    }
    if w.any(&["no vanes", "without vanes"]) {
        Some(ElbowMitred)
    } else if w.any(&["vane", "vanes", "vaned"]) {
        Some(ElbowMitredVanes)
    } else if w.any(&["mitred", "mitered", "mitre", "miter", "square", "sharp"]) {
        Some(ElbowMitred)
    } else if w.any(&["radius", "radiused", "smooth", "gored", "pleated", "segmented", "round", "d", "r w"]) {
        Some(ElbowRadius)
    } else {
        None
    }
}

/// The element type, or `None` when the name does not settle it.
pub fn classify(category: Category, family: &str, type_name: &str) -> Option<ElementType> {
    let w = Words::new(&format!("{family} {type_name}"));
    let fits = |cats: &[Category]| category == Category::Other || cats.contains(&category);
    for (cats, phrases, ty) in RULES {
        if fits(cats) && w.any(phrases) {
            return *ty;
        }
    }
    if fits(INLINE) {
        if let Some(ty) = unit(&w) {
            return ty;
        }
    }
    for (cats, phrases, ty) in OUTLETS {
        if fits(cats) && w.any(phrases) {
            return *ty;
        }
    }
    if !fits(FITTING) {
        return None;
    }
    if let Some(ty) = elbow(&w) {
        return Some(ty);
    }
    if w.any(&["endcap", "end cap", "cap", "blank", "blanking"]) {
        Some(EndCap)
    } else if w.any(&["union", "coupling", "connector", "joint", "access door", "access panel"]) {
        Some(Coupling)
    } else {
        None
    }
}

/// The IFC export's own class, where it is specific enough to trust. Its
/// bends take their qualifier from the type description ("smooth, R=D").
fn from_ifc(ifc: &IfcInfo) -> Option<ElementType> {
    match (ifc.class.as_str(), ifc.predefined.as_str()) {
        ("IFCDUCTSILENCER", _) => Some(Silencer),
        ("IFCDAMPER", "FIREDAMPER" | "SMOKEDAMPER" | "FIRESMOKEDAMPER") => Some(FireSmokeDamper),
        ("IFCDAMPER", "BALANCINGDAMPER" | "CONTROLDAMPER" | "BACKDRAFTDAMPER" | "GRAVITYDAMPER" | "RELIEFDAMPER") => Some(VolumeDamper),
        ("IFCAIRTERMINALBOX", _) => Some(VavTerminalUnit),
        ("IFCAIRTOAIRHEATRECOVERY", _) => Some(AirHandlingUnit),
        ("IFCDUCTFITTING", "TRANSITION") => Some(Transition),
        ("IFCDUCTFITTING", "BEND") => elbow(&Words::new(&format!("bend {}", ifc.description))),
        _ => None,
    }
}

/// A node's type: the name rules (with the IFC description, where merged),
/// then the IFC class.
pub fn classify_node(node: &Node) -> Option<ElementType> {
    let description = node.ifc.as_ref().map_or("", |i| i.description.as_str());
    classify(Category::of(&node.kind), &node.family, &format!("{} {description}", node.type_name))
        .or_else(|| node.ifc.as_ref().and_then(from_ifc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specific_names_beat_general_ones() {
        let c = |cat, family| classify(cat, family, "").map(ElementType::label);
        assert_eq!(c(Equipment, "Fan Powered Terminal Unit - Series"), Some("vav_terminal_unit"));
        assert_eq!(c(Fitting, "Duct Silencer Elbow"), Some("silencer"));
        assert_eq!(c(Fitting, "Combination Fire/Smoke Damper"), Some("fire_smoke_damper"));
        assert_eq!(c(Terminal, "M_Supply Diffuser - Sidewall"), Some("grille_or_register"));
        assert_eq!(c(Terminal, "Air Terminal-Supply Cap-FB"), Some("external_cap_or_louvre"));
        assert_eq!(c(Fitting, "Round Endcap"), Some("end_cap"));
        assert_eq!(c(Fitting, "XX_Duct_Bend_Square_NoVanes"), Some("elbow_mitred"));
        assert_eq!(c(Equipment, "Heat Pump"), None);
        assert_eq!(c(Equipment, "Fan Coil Unit"), Some("fan_coil_or_heat_pump"));
        assert_eq!(c(Equipment, "Tank Exhaust Fan"), None);
        assert_eq!(c(Fitting, "Rectangular Elbow"), None);
    }
}
