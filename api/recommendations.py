"""
SCI-RTW Predictor: confidence level and suggested considerations
Implements the "confidence level" and "recommendations" parts of the
prediction results page (proposal Fig. 3.13) and the recommendation notes
of the Prediction Report entity (Table 3.10).

Scope, per proposal section 1.4 and 3.6.4: the system does not give
treatment recommendations, diagnoses or rehabilitation prescriptions. What
is generated here are planning prompts for the care team - which kind of
support to consider discussing - chosen by fixed, readable rules from the
prediction and from the factors that lowered this patient's estimate. They
are not produced by the model and are not medical advice; the clinician's
own notes sit alongside them.
"""

# Fields grouped by the kind of support that addresses them. Only a field
# that LOWERED this patient's estimate can trigger its suggestion.
INJURY_FUNCTION_FIELDS = ("asia_motor_total_discharge", "aisgrade_discharge",
                          "injury_completeness_discharge", "neuro_level_discharge")
# Ignore contributions too small to be worth acting on (log-odds units).
MIN_CONTRIBUTION = 0.05
MAX_SUGGESTIONS = 4


def confidence_for(probability: float, bands: list) -> dict:
    """Confidence level from how far the estimate is from 50%, with the
    accuracy measured for that level on held-out test patients."""
    distance = abs(probability - 0.5)
    band = [b for b in bands if distance >= b["min_distance"]][-1]
    return {
        "level": band["level"],
        "test_accuracy": band["accuracy"],
        "n_test": band["n_test"],
        # Where each level starts, as distance from 50% - lets the results
        # page draw the confidence zones on its probability scale.
        "thresholds": {b["level"]: b["min_distance"] for b in bands},
        "basis": (f"On held-out test patients whose estimate was this far from 50%, "
                  f"the prediction was correct {band['accuracy']:.0%} of the time "
                  f"({band['n_test']:,} patients)."),
    }


def suggest(rtw_status: str, confidence_level: str, rtw_timing, contributions_by_field: dict) -> list:
    """Rule-based planning prompts. Each has a title, the text, and the
    reason it was shown, so the clinician can see why it appeared."""
    lowering = {f: c for f, c in contributions_by_field.items() if c <= -MIN_CONTRIBUTION}
    out = []

    # 1. Overall direction, tempered by confidence.
    if confidence_level == "Low":
        out.append({
            "title": "Treat this estimate as uncertain",
            "text": "The estimate is close to 50%, where the model is right only a little more often than "
                    "chance. Base planning on the clinical assessment and revisit after further rehabilitation.",
            "reason": "Low confidence level.",
        })
    elif rtw_status == "RTW":
        out.append({
            "title": "Begin return-to-work planning early",
            "text": "Consider raising work goals in rehabilitation planning now, including early contact with "
                    "the employer about duties and adjustments.",
            "reason": f"Return to work is the more likely outcome ({confidence_level.lower()} confidence).",
        })
    else:
        out.append({
            "title": "Consider early vocational rehabilitation referral",
            "text": "A lower estimate identifies a patient who may benefit most from additional vocational "
                    "support. It is not a verdict that return to work cannot happen.",
            "reason": f"Return to work is the less likely outcome ({confidence_level.lower()} confidence).",
        })

    # 2. Support matched to what lowered this patient's estimate, largest first.
    specific = []
    if "education_level" in lowering:
        specific.append((lowering["education_level"], {
            "title": "Discuss education and skills training",
            "text": "Consider vocational counselling on training or further education, which can widen the "
                    "range of suitable jobs.",
            "reason": "Education level lowered this patient's estimate.",
        }))
    if "pre_injury_employment_status" in lowering:
        specific.append((lowering["pre_injury_employment_status"], {
            "title": "Assess work history and job-seeking support",
            "text": "Consider a vocational assessment and supported job-seeking, as there may be no existing "
                    "job or employer to return to.",
            "reason": "Pre-injury employment status lowered this patient's estimate.",
        }))
    injury = [lowering[f] for f in INJURY_FUNCTION_FIELDS if f in lowering]
    if injury:
        specific.append((sum(injury), {
            "title": "Plan workplace accommodations",
            "text": "Consider an assessment of job modification, assistive technology and workplace access "
                    "alongside continued functional goals.",
            "reason": "Injury severity or motor function lowered this patient's estimate.",
        }))
    if "age_at_injury" in lowering:
        specific.append((lowering["age_at_injury"], {
            "title": "Discuss options suited to career stage",
            "text": "Consider discussing phased return, role change or retraining in light of the patient's "
                    "remaining working years and preferences.",
            "reason": "Age at injury lowered this patient's estimate.",
        }))
    out.extend(item for _, item in sorted(specific, key=lambda pair: pair[0]))

    # 3. Timing, only when return to work is predicted.
    if rtw_status == "RTW" and rtw_timing == "Year 1":
        out.append({
            "title": "Prepare for a return within the first year",
            "text": "Consider agreeing a work plan before discharge and a review in the first months after.",
            "reason": "Most likely timing is Year 1. Timing estimates are less reliable than the RTW estimate.",
        })
    elif rtw_status == "RTW" and rtw_timing is not None:
        out.append({
            "title": "Keep work on the agenda at follow-up",
            "text": "Return may take several years. Consider reviewing work goals at each follow-up visit so "
                    "support continues beyond initial rehabilitation.",
            "reason": f"Most likely timing is {rtw_timing}. Timing estimates are less reliable than the RTW estimate.",
        })

    return out[:MAX_SUGGESTIONS]
