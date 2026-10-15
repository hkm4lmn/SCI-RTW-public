import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { getLabel } from "../constants/labelMaps.js";
import { BarList, ColumnChart, SplitBar } from "./charts.jsx";

const WEEKS_SHOWN = 8;
const DAY_MS = 24 * 60 * 60 * 1000;
const CONFIDENCE_LEVELS = ["High", "Moderate", "Low"];
const CONFIDENCE_TONES = { High: "strong", Moderate: "medium", Low: "light" };

// Saved assessments per week for the last WEEKS_SHOWN weeks, oldest first.
function weeklyCounts(history) {
  const end = new Date();
  end.setHours(23, 59, 59, 999);
  const weeks = Array.from({ length: WEEKS_SHOWN }, (_, i) => {
    const to = new Date(end.getTime() - (WEEKS_SHOWN - 1 - i) * 7 * DAY_MS);
    const from = new Date(to.getTime() - 7 * DAY_MS + 1);
    return { from, to, value: 0 };
  });
  history.forEach((h) => {
    const t = new Date(h.timestamp).getTime();
    const week = weeks.find((w) => t >= w.from.getTime() && t <= w.to.getTime());
    if (week) week.value += 1;
  });
  const short = (d) => d.toLocaleDateString(undefined, { day: "numeric", month: "short" });
  return weeks.map((w) => ({
    label: short(w.from),
    title: `Week of ${short(w.from)} to ${short(w.to)}`,
    value: w.value,
  }));
}

export default function Dashboard({ history, onNavigate }) {
  const [modelInfo, setModelInfo] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.modelInfo().then(setModelInfo).catch((e) => setError(e.message));
  }, []);

  const total = history.length;
  const rtwCount = history.filter((h) => h.result.rtw_status === "RTW").length;
  const patientCount = new Set(history.map((h) => h.patient.id)).size;
  const l1 = modelInfo?.layer1?.models_compared?.xgboost;
  const bands = modelInfo?.layer1?.evaluation?.confidence_bands || [];

  const confidenceItems = CONFIDENCE_LEVELS.map((level) => {
    const band = bands.find((b) => b.level === level);
    return {
      label: level,
      value: history.filter((h) => h.result.confidence?.level === level).length,
      tone: CONFIDENCE_TONES[level],
      note: band ? `right ${Math.round(band.accuracy * 100)}% of the time in testing` : null,
    };
  });

  return (
    <div>
      <div className="page-header with-action">
        <div>
          <h2>Clinical Prediction Dashboard</h2>
          <p>Return-to-work predictions for spinal cord injury patients, from all saved assessments.</p>
        </div>
        <button className="btn primary" onClick={() => onNavigate("predict")}>New prediction</button>
      </div>

      <div className="disclaimer">
        This tool is designed to assist clinical decision-making and should not replace
        professional medical judgment. Predictions should be interpreted alongside a
        comprehensive patient assessment.
      </div>

      <div className="stat-row">
        <div className="stat-card">
          <div className="label">Saved assessments</div>
          <div className="value">{total}</div>
        </div>
        <div className="stat-card">
          <div className="label">Patients assessed</div>
          <div className="value">{total ? patientCount : "—"}</div>
        </div>
        <div className="stat-card">
          <div className="label">Positive RTW predictions</div>
          <div className="value">{total ? rtwCount : "—"}</div>
        </div>
        <div className="stat-card">
          <div className="label">Model test accuracy (Layer 1)</div>
          <div className="value">{l1 ? `${Math.round(l1.accuracy * 100)}%` : "…"}</div>
        </div>
      </div>

      {error && <div className="error-banner">Couldn't load model info: {error}</div>}

      {total === 0 ? (
        <div className="panel">
          <div className="empty-state">
            <p>No assessments have been saved yet.</p>
            <button className="btn primary" onClick={() => onNavigate("predict")}>Run the first prediction</button>
          </div>
        </div>
      ) : (
        <>
          <div className="panel-grid">
            <div className="panel">
              <h3>Predicted outcomes</h3>
              <p className="sub">Share of saved assessments predicted to return to work within 10 years.</p>
              <SplitBar parts={[
                { label: "Likely to return", value: rtwCount, tone: "rtw" },
                { label: "Unlikely to return", value: total - rtwCount, tone: "non-rtw" },
              ]} />
            </div>
            <div className="panel">
              <h3>How confident the predictions are</h3>
              <p className="sub">Saved assessments by confidence level.</p>
              <BarList items={confidenceItems} />
            </div>
          </div>

          <div className="panel">
            <h3>Assessments per week</h3>
            <p className="sub">Saved assessments over the last {WEEKS_SHOWN} weeks. Each label is the first day of the week.</p>
            <ColumnChart points={weeklyCounts(history)} />
          </div>

          <div className="panel">
            <div className="panel-toolbar">
              <div>
                <h3>Recent assessments</h3>
                <p className="sub" style={{ margin: 0 }}>Last {Math.min(total, 5)} saved predictions.</p>
              </div>
              <button className="btn ghost" onClick={() => onNavigate("history")}>View all history</button>
            </div>
            <div className="table-scroll">
              <table className="history">
                <thead>
                  <tr>
                    <th>Patient</th>
                    <th>Age at injury</th>
                    <th>AIS Grade</th>
                    <th>RTW Prediction</th>
                    <th>Probability</th>
                    <th>Confidence</th>
                    <th>Timing</th>
                  </tr>
                </thead>
                <tbody>
                  {history.slice(0, 5).map((h) => (
                    <tr key={h.id}>
                      <td>{h.patient.patient_code}</td>
                      <td>{h.input.age_at_injury}</td>
                      <td>{getLabel("aisgrade_discharge", h.input.aisgrade_discharge)}</td>
                      <td>
                        <span className={`pill ${h.result.rtw_status === "RTW" ? "rtw" : "non-rtw"}`}>
                          {h.result.rtw_status}
                        </span>
                      </td>
                      <td>{Math.round(h.result.rtw_probability * 100)}%</td>
                      <td>{h.result.confidence?.level || "—"}</td>
                      <td>{h.result.rtw_timing || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
