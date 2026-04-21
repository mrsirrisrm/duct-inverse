// Gemini call — browser-side. Key in localStorage; the user pastes once and
// the app hits the Gemini REST endpoint directly. Same pattern as the
// partition web app. Prompt is the duct advisor's, mirroring the Python
// PoC so the advice is consistent between the two tools.

const SYSTEM_PROMPT = `You are an experienced mechanical-services acoustic consultant advising an engineer who is laying out a small HVAC supply path from a source (FCU or similar) to a room diffuser. Given a JSON object describing the current system — block-by-block design, receiver octave-band SPL, NC rating, controlling bands, and top sensitivities — write SHORT, actionable design advice.

Rules:
- Identify the weakest point in the current design using "controlling_bands" (the octave bands exceeding the NC target curve) and the block that is most responsible (infer from block types in "blocks" and the ranking in "top_sensitivities").
- Recommend the single most impactful change. Sensitivities are dSPL / d(parameter) weighted toward the controlling bands, so a large-magnitude negative value means increasing that parameter lowers NC.
- Use HVAC acoustics vocabulary (regenerative noise, duct lining, insertion loss, splitter attenuator, end reflection, room constant) but be concrete: name the block and name the parameter.
- Keep it to 3 to 5 sentences. Plain prose, no bullet points.
- Don't restate the NC number — it's shown separately in the UI.
- Do not invent products, codes, or numbers not present in the JSON.`;

const DEFAULT_MODEL = "gemini-3.1-flash-lite-preview";
const KEY_STORAGE = "wall-duct-gemini-key";

export function getApiKey() {
  try {
    return localStorage.getItem(KEY_STORAGE) || "";
  } catch {
    return "";
  }
}

export function setApiKey(k) {
  try {
    if (k) localStorage.setItem(KEY_STORAGE, k);
    else localStorage.removeItem(KEY_STORAGE);
  } catch {}
}

export async function adviceFor(facts, constraints = "", model = DEFAULT_MODEL) {
  const key = getApiKey();
  if (!key) throw new Error("No API key set.");
  const constraintsBlock = constraints.trim()
    ? `\n\nAdditional constraints the user has given (respect these — do not recommend a change that violates them):\n${constraints.trim()}`
    : "";
  const prompt = `Here is the current duct-system analysis:\n\n\`\`\`json\n${JSON.stringify(facts, null, 2)}\n\`\`\`${constraintsBlock}\n\nWrite the advice now.`;

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(key)}`;
  const body = {
    systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
    contents: [{ role: "user", parts: [{ text: prompt }] }],
    generationConfig: { temperature: 0.3 },
  };

  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`Gemini ${resp.status}: ${txt.slice(0, 300)}`);
  }
  const data = await resp.json();
  const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) throw new Error("Empty response from Gemini.");
  return text.trim();
}
