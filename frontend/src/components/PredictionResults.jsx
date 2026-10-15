import React from "react";
import { FIELD_LABELS, describeValue } from "../constants/labelMaps.js";
import { ProbabilityScale } from "./charts.jsx";

const TIMING_ORDER = ["Year 1", "Year 5", "By Year 10"];

const formatDate = (d) => d.toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" });

export default function PredictionResults({
  result, patient, onNewPrediction, actionLabel = "New Assessment", assessedAt, enteredBy, children,
}) {
  const isRtw = result.rtw_status === "RTW";
  // Chronological, so the three bars read as a timeline.
  const timingEntries = result.rtw_timing_probabilities
    ? TIMING_ORDER.filter((t) => t in result.rtw_timing_probabilities)
        .map((t) => [t, result.rtw_timing_probabilities[t]])
    : [];
  const scaleBands = result.confidence?.thresholds
    ? Object.entries(result.confidence.thresholds).map(([level, min_distance]) => ({ level, min_distance }))
    : null;
  const factors = result.patient_factors || [];
  const maxMag = Math.max(...factors.map((f) => Math.abs(f.contribution)), 0.0001);
  const confidence = result.confidence;
  const suggestions = result.recommendations || [];

  return (
    <div>
      <div className="page-header with-action no-print">
        <div>
          <h2>Prediction Results</h2>
          <p>
            {patient
              ? `RTW probability and timeline assessment for ${patient.full_name} (${patient.patient_code}).`
              : "RTW probability and timeline assessment based on the entered patient profile."}
          </p>
        </div>
        <button className="btn ghost" onClick={() => window.print()}>Print report</button>
      </div>

      {/* Heading of the printed report; the screen heading above is not printed. */}
      <div className="print-only report-header">
        <h2>Return-to-work prediction report</h2>
        <dl>
          {patient && <><dt>Patient</dt><dd>{patient.full_name} ({patient.patient_code})</dd></>}
          <dt>Assessed</dt><dd>{formatDate(assessedAt ? new Date(assessedAt) : new Date())}</dd>
          {enteredBy && <><dt>Entered by</dt><dd>{enteredBy}</dd></>}
          <dt>Printed</dt><dd>{formatDate(new Date())}</dd>
        </dl>
      </div>

      {result.from_current_model === false && (
        <div className="disclaimer">
          This assessment was produced by an earlier version of the model. The same details
          entered today may give a different estimate.
        </div>
      )}

      <div className={`result-badge ${isRtw ? "rtw" : "non-rtw"}`}>
        <div className="result-summary">
          <div>
            <div className="label">RTW prediction</div>
            <div className="status">{isRtw ? "Likely to return to work" : "Unlikely to return to work"}</div>
            <div className="prob">
              Estimated RTW probability: <strong>{Math.round(result.rtw_probability * 100)}%</strong>
            </div>
          </div>
          {confidence && (
            <div className="confidence">
              <div className="prob">
                Confidence level: <strong>{confidence.level}</strong>
              </div>
              <div className="basis">{confidence.basis}</div>
            </div>
          )}
        </div>
        <ProbabilityScale probability={result.rtw_probability} bands={scaleBands} />
      </div>

      {isRtw && timingEntries.length > 0 && (
        <div className="panel">
          <h3>Estimated timing of first return to work</h3>
          <p className="sub">
            If this patient returns to work, which follow-up is most likely to be the first
            to find them working. The three chances add up to 100%. Timing estimates are
            less reliable than the RTW estimate above.
          </p>
          <div className="timing-bars">
            {timingEntries.map(([label, prob]) => (
              <div className={`timing-bar-row ${label === result.rtw_timing ? "most-likely" : ""}`} key={label}>
                <span className="name">{label}</span>
                <div className="timing-bar-track">
                  <div className="timing-bar-fill" style={{ width: `${prob * 100}%` }} />
                </div>
                <span className="pct">{Math.round(prob * 100)}%</span>
                <span className="tag">{label === result.rtw_timing ? "Most likely" : ""}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="panel">
        <h3>What drove this prediction</h3>
        <p className="sub">
          How each entered value moved this patient's estimate up or down (XGBoost per-prediction
          contributions). These reflect statistical patterns in the training data, not causes.
        </p>
        {factors.map((f) => {
          const up = f.effect === "increases";
          return (
            <div className="factor-row" key={f.field}>
              <span className="fname">
                <strong>{FIELD_LABELS[f.field] || f.field}</strong>
                <span className="fvalue"> — {describeValue(f.field, f.value)}</span>
              </span>
              <span className={`effect ${up ? "up" : "down"}`}>
                {up ? "Raises" : "Lowers"} RTW likelihood
              </span>
              <div className="factor-bar-track">
                <div
                  className={`factor-bar-fill ${up ? "up" : "down"}`}
                  style={{ width: `${(Math.abs(f.contribution) / maxMag) * 100}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {suggestions.length > 0 && (
        <div className="panel">
          <h3>Suggested considerations</h3>
          <p className="sub">
            Planning prompts for the care team, selected by fixed rules from this result. They
            are not treatment recommendations and do not replace clinical judgment.
          </p>
          {suggestions.map((s) => (
            <div className="suggestion" key={s.title}>
              <strong>{s.title}</strong>
              <p>{s.text}</p>
              <span className="why">Shown because: {s.reason}</span>
            </div>
          ))}
        </div>
      )}

      <div className="disclaimer">
        This prediction is a decision-support estimate, not a diagnosis or guarantee of
        outcome. It should be interpreted by a qualified clinician alongside a full
        patient assessment.
      </div>

      {children}

      <button className="btn primary no-print" onClick={onNewPrediction}>{actionLabel}</button>
    </div>
  );
}
