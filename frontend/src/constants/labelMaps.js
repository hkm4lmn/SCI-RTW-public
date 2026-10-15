// Label lookups sourced from Data_Dictionary_Public_AR2021.pdf (NSCISC).
// IMPORTANT: these maps are for DISPLAY ONLY. The actual valid dropdown
// *values* always come live from GET /category-options (the exact strings
// the trained model's encoder expects) — see api.js + PredictionForm.jsx.
// getLabel() looks up a friendly label for whatever value the API returns,
// and falls back gracefully for anything not explicitly mapped.
//
// NOTE: most codes are keyed with a trailing ".0" (e.g. "1.0" not "1").
// This isn't a style choice -- several source columns (sex, education,
// marital status, injury cause, veteran status, vertebral/associated
// injury, work-related injury) only picked up decimals because at least
// one row in that column became NaN during cleaning, which upcasts the
// whole pandas column from int64 to float64. aisgrade_discharge and
// neuro_level_discharge were never purely numeric, so they keep integer-
// looking / alphanumeric codes with no decimal. Always check
// category_options.json (or GET /category-options) rather than assume.
//
// race, hispanic_origin, pre_injury_job_class, and depression_severity_phq
// are intentionally absent: all four were removed from the trained model
// after testing (see docs/test_report.md) -- race/hispanic_origin gave no
// accuracy benefit, PHQ and job-class acted as injury-era proxies rather
// than clinical signals.

export const LABEL_MAPS = {
  sex: { "1.0": "Male", "2.0": "Female", "3.0": "Other, Transgender" },

  education_level: {
    "1.0": "8th grade or less", "2.0": "9th–11th grade",
    "3.0": "High School Diploma or G.E.D.", "4.0": "Associate Degree",
    "5.0": "Bachelor Degree", "6.0": "Master Degree",
    "7.0": "Doctorate (Ph.D., M.D., etc.)", "8.0": "Other, unclassified",
  },

  marital_status: {
    "1.0": "Never Married (Single)", "2.0": "Married", "3.0": "Divorced",
    "4.0": "Separated", "5.0": "Widowed", "6.0": "Other, unclassified",
    "7.0": "Living with Significant Other / Partner",
  },

  veteran_status: { "0.0": "No", "1.0": "Yes" },

  work_related_injury: { "0.0": "No", "1.0": "Yes" },

  vertebral_injury_level: { "0.0": "No fracture/dislocation", "1.0": "Yes, fracture/dislocation present" },

  associated_injury: { "0.0": "No associated injury", "1.0": "Yes, associated injury present" },

  pre_injury_employment_status: {
    "1.0": "Working (competitive labor market)", "2.0": "Homemaker",
    "3.0": "On-the-job training", "4.0": "Sheltered workshop",
    "5.0": "Retired (legacy code)", "6.0": "Student", "7.0": "Unemployed",
    "8.0": "Retired, disability pension", "9.0": "Retired, non-disability",
    "10.0": "Other, unclassified",
  },

  injury_cause: {
    "1.0": "Auto accident", "2.0": "Motorcycle accident", "3.0": "Other vehicular, unclassified",
    "4.0": "Boat", "5.0": "Fixed-wing aircraft", "6.0": "Rotating wing aircraft (helicopter)",
    "7.0": "Snowmobile", "8.0": "Bicycle", "9.0": "All-terrain vehicle (ATV/ATC)",
    "10.0": "Gunshot wound", "11.0": "Other penetrating wound (stabbing, etc.)",
    "12.0": "Person-to-person contact (violence)", "15.0": "Explosion",
    "20.0": "Diving", "21.0": "Football", "22.0": "Trampoline", "23.0": "Snow skiing",
    "24.0": "Water skiing", "25.0": "Other sport, unclassified", "26.0": "Wrestling",
    "27.0": "Baseball/softball", "28.0": "Basketball/volleyball", "29.0": "Surfing",
    "70.0": "Horseback riding", "71.0": "Gymnastics", "72.0": "Rodeo",
    "73.0": "Track and field", "74.0": "Field sports (soccer, rugby, etc.)",
    "75.0": "Hang gliding", "76.0": "Air sports (parachuting, etc.)",
    "77.0": "Winter sports (sledding, snowboarding, etc.)", "78.0": "Skateboard",
    "30.0": "Fall", "31.0": "Hit by falling/flying object",
    "40.0": "Pedestrian struck by vehicle",
    "50.0": "Medical/surgical complication",
    "60.0": "Other, unclassified",
  },

  aisgrade_discharge: {
    "A": "A — Complete (no sensory/motor in S4-S5)",
    "B": "B — Incomplete (sensory only preserved)",
    "C": "C — Incomplete (motor preserved, most muscles <3)",
    "D": "D — Incomplete (motor preserved, most muscles ≥3)",
    "E": "E — Normal",
    // Legacy Frankel Grade numeric codes. No longer offered by the form: the
    // pipeline maps each to its AIS letter (see data_pipeline.py). Kept so a
    // prediction saved before that change still shows what was entered.
    "1": "Incomplete – Sensation Only (legacy Frankel B)",
    "2": "Incomplete – Motor, Non-functional (legacy Frankel C)",
    "3": "Incomplete – Motor, Functional (legacy Frankel D)",
    "4": "Complete Recovery (legacy Frankel E)",
    "5": "Complete (legacy Frankel A)",
  },

  injury_completeness_discharge: {
    "1.0": "Paraplegia, incomplete", "2.0": "Paraplegia, complete",
    "3.0": "Paraplegia, minimal deficit", "4.0": "Tetraplegia, incomplete",
    "5.0": "Tetraplegia, complete", "6.0": "Tetraplegia, minimal deficit",
    "7.0": "Normal neurologic",
    "8.0": "Normal neurologic, minimal deficit (legacy code)",
  },
};

// Neurologic level codes follow a structured pattern (Region + 2-digit segment,
// e.g. "C04", "T12", "L01", "S02") plus a fixed special code, so this is
// generated rather than hand-mapped -- it correctly labels any legacy code
// (e.g. "S04") the API returns without needing an exhaustive list.
const NEURO_REGION_NAMES = { C: "Cervical", T: "Thoracic", L: "Lumbar", S: "Sacral" };

export function neuroLevelLabel(code) {
  if (code === "X00") return "Normal neurologic (no deficit)";
  const region = code[0];
  const rest = code.slice(1);
  const regionName = NEURO_REGION_NAMES[region];
  if (!regionName) return code;
  if (rest === "99") return `${regionName}, Unknown Level`;
  const segment = parseInt(rest, 10);
  return `${regionName} ${region}${String(segment).padStart(2, "0")}`;
}

// Generic label lookup: uses the mapped label if available, otherwise falls
// back to the raw code rather than hiding the option.
export function getLabel(field, value) {
  if (field === "neuro_level_discharge") return neuroLevelLabel(value);
  const map = LABEL_MAPS[field];
  if (map && map[value] !== undefined) return map[value];
  return value;
}

// Human-readable field names (for factor lists, tables, tooltips).
export const FIELD_LABELS = {
  sex: "Biological sex",
  age_at_injury: "Age at injury",
  education_level: "Education level",
  marital_status: "Marital status",
  veteran_status: "Veteran status",
  pre_injury_employment_status: "Pre-injury employment status",
  injury_cause: "Cause of injury",
  vertebral_injury_level: "Vertebral fracture/dislocation",
  associated_injury: "Associated injury",
  work_related_injury: "Work-related injury",
  neuro_level_discharge: "Neurologic level (discharge)",
  aisgrade_discharge: "AIS grade (discharge)",
  injury_completeness_discharge: "Injury completeness",
  asia_motor_total_discharge: "ASIA motor total score",
};

// Human-readable text for a value the user entered for `field`.
export function describeValue(field, value) {
  if (value === null || value === undefined || value === "") return "not assessed";
  if (LABEL_MAPS[field] || field === "neuro_level_discharge") return getLabel(field, String(value));
  return String(value);
}

// Turns an encoded model feature name (e.g. "pre_injury_employment_status_5.0")
// into readable text.
const NUMERIC_FIELDS = ["age_at_injury", "asia_motor_total_discharge"];
const CAT_FIELDS_LONGEST_FIRST = Object.keys(LABEL_MAPS)
  .concat(["neuro_level_discharge"])
  .sort((a, b) => b.length - a.length);

export function describeFeature(feature) {
  for (const f of NUMERIC_FIELDS) {
    if (feature === f) return FIELD_LABELS[f];
  }
  for (const f of CAT_FIELDS_LONGEST_FIRST) {
    if (feature.startsWith(`${f}_`)) {
      const value = feature.slice(f.length + 1);
      return `${FIELD_LABELS[f]}: ${getLabel(f, value)}`;
    }
  }
  return feature;
}
