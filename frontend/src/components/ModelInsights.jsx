import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { describeFeature, FIELD_LABELS } from "../constants/labelMaps.js";
import { IntervalPlot } from "./charts.jsx";

const MODEL_LABELS = {
  xgboost: "XGBoost (main)",
  random_forest: "Random Forest (comparison)",
  logistic_regression: "Logistic Regression (baseline)",
};

const MODEL_ORDER = ["xgboost", "random_forest", "logistic_regression"];

function bestModel(models, key) {
  return Object.entries(models).reduce((best, [name, m]) => (m[key] > (best?.[1]?.[key] ?? -1) ? [name, m] : best), null)?.[0];
}

function LayerTable({ layer }) {
  const models = layer.models_compared;
  const isBinary = "roc_auc" in Object.values(models)[0];
  const accKey = "accuracy";
  const f1Key = isBinary ? "f1_score" : "f1_macro";
  const aucKey = isBinary ? "roc_auc" : "roc_auc_ovr";
  const bestAcc = bestModel(models, accKey);
  const bestAuc = bestModel(models, aucKey);

  return (
    <table className="metrics">
      <thead>
        <tr>
          <th>Model</th>
          <th>Accuracy</th>
          <th>{isBinary ? "Precision" : "Precision (macro)"}</th>
          <th>{isBinary ? "Recall" : "Recall (macro)"}</th>
          <th>{isBinary ? "F1" : "F1 (macro)"}</th>
          <th>ROC-AUC</th>
        </tr>
      </thead>
      <tbody>
        {MODEL_ORDER.filter((n) => models[n]).map((n) => [n, models[n]]).map(([name, m]) => (
          <tr key={name}>
            <td>{MODEL_LABELS[name] || name}</td>
            <td className={name === bestAcc ? "best" : ""}>{(m[accKey] * 100).toFixed(1)}%</td>
            <td>{((isBinary ? m.precision : m.precision_macro) * 100).toFixed(1)}%</td>
            <td>{((isBinary ? m.recall : m.recall_macro) * 100).toFixed(1)}%</td>
            <td>{(m[f1Key] * 100).toFixed(1)}%</td>
            <td className={name === bestAuc ? "best" : ""}>{m[aucKey].toFixed(3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function FeatureImportance({ features }) {
  const max = Math.max(...features.map((f) => f.importance), 0.0001);
  return (
    <div>
      {features.map((f) => (
        <div className="factor-row" key={f.feature}>
          <span className="fname">{describeFeature(f.feature)}</span>
          <div className="factor-bar-track">
            <div className="factor-bar-fill" style={{ width: `${(f.importance / max) * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

const pct = (v) => `${(v * 100).toFixed(1)}%`;

// Cross-validation, overfitting check and test-set uncertainty (proposal 3.6.1).
function ValidationTable({ evaluation }) {
  return (
    <table className="metrics">
      <thead>
        <tr>
          <th>Model</th>
          <th>5-fold CV accuracy</th>
          <th>Training accuracy</th>
          <th>Test accuracy</th>
          <th>Train − test gap</th>
          <th>Test accuracy 95% interval</th>
        </tr>
      </thead>
      <tbody>
        {MODEL_ORDER.filter((n) => evaluation.cross_validation[n]).map((name) => {
          const cv = evaluation.cross_validation[name].accuracy;
          const tv = evaluation.train_vs_test[name];
          const ci = evaluation.test_ci[name].accuracy;
          return (
            <tr key={name}>
              <td>{MODEL_LABELS[name] || name}</td>
              <td>{pct(cv.mean)} ± {(cv.std * 100).toFixed(1)}</td>
              <td>{pct(tv.train_accuracy)}</td>
              <td>{pct(tv.test_accuracy)}</td>
              <td>{(tv.gap * 100).toFixed(1)} pts</td>
              <td>{pct(ci[0])} – {pct(ci[1])}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// One bar per input field: how much the score drops when that field is
// shuffled in the held-out test set (permutation importance).
function FactorImportance({ factors }) {
  const max = Math.max(...factors.fields.map((f) => f.auc_drop_mean), 0.0001);
  return (
    <div>
      {factors.fields.map((f) => (
        <div className="factor-row" key={f.field}>
          <span className="fname">{FIELD_LABELS[f.field] || f.field}</span>
          <span className="effect">{f.auc_drop_mean > 0 ? `−${f.auc_drop_mean.toFixed(3)} AUC` : "no measurable effect"}</span>
          <div className="factor-bar-track">
            <div className="factor-bar-fill" style={{ width: `${(Math.max(f.auc_drop_mean, 0) / max) * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function PerClassTable({ perClass, labels }) {
  return (
    <table className="metrics">
      <thead>
        <tr>
          <th>Timing class</th>
          <th>Test patients</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1</th>
        </tr>
      </thead>
      <tbody>
        {labels.filter((l) => perClass[l]).map((label) => (
          <tr key={label}>
            <td>{label}</td>
            <td>{perClass[label].support}</td>
            <td>{pct(perClass[label].precision)}</td>
            <td>{pct(perClass[label].recall)}</td>
            <td>{pct(perClass[label].f1)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const TIMING_ORDER = ["Year 1", "Year 5", "By Year 10"];

export default function ModelInsights() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState("layer1");

  useEffect(() => {
    api.modelInfo().then(setInfo).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="error-banner">Couldn't load model insights: {error}</div>;
  if (!info) return <p className="loading-note">Loading model insights…</p>;

  const layer = tab === "layer1" ? info.layer1 : info.layer2;
  const evaluation = layer.evaluation;

  return (
    <div>
      <div className="page-header">
        <h2>Model Insights &amp; Performance</h2>
        <p>Comparison of the three trained models and feature importance, per layer.</p>
      </div>

      <div className="model-tabs">
        <button className={`model-tab ${tab === "layer1" ? "active" : ""}`} onClick={() => setTab("layer1")}>
          Layer 1 — RTW Status
        </button>
        <button className={`model-tab ${tab === "layer2" ? "active" : ""}`} onClick={() => setTab("layer2")}>
          Layer 2 — RTW Timing
        </button>
      </div>

      <div className="panel">
        <h3>Model comparison</h3>
        <p className="sub">Held-out test set. Best value in each ranking column is highlighted.</p>
        {evaluation && (
          <IntervalPlot
            min={0.45} max={0.75}
            reference={{
              value: evaluation.majority_baseline.accuracy,
              label: `always predicting "${evaluation.majority_baseline.always_predicts}"`,
            }}
            rows={MODEL_ORDER.filter((n) => layer.models_compared[n]).map((n) => ({
              label: MODEL_LABELS[n],
              value: layer.models_compared[n].accuracy,
              low: evaluation.test_ci[n].accuracy[0],
              high: evaluation.test_ci[n].accuracy[1],
            }))}
          />
        )}
        <div className="table-scroll"><LayerTable layer={layer} /></div>
        {evaluation && (
          <p className="sub" style={{ margin: "14px 0 0" }}>
            Reference point: always predicting "{evaluation.majority_baseline.always_predicts}" scores{" "}
            {pct(evaluation.majority_baseline.accuracy)} accuracy on this test set
            ({evaluation.n_test.toLocaleString()} patients; models trained on {evaluation.n_train.toLocaleString()}).
          </p>
        )}
      </div>

      {evaluation && (
        <div className="panel">
          <h3>Validation</h3>
          <p className="sub">
            Cross-validation on the training data, and training versus test accuracy as an
            overfitting check. Differences between models that fall inside the 95% interval
            are not meaningful.
          </p>
          <ValidationTable evaluation={evaluation} />
        </div>
      )}

      {evaluation?.confidence_bands && (
        <div className="panel">
          <h3>What the confidence level means</h3>
          <p className="sub">
            Each prediction is labelled by how far its estimate is from 50%. This is how often
            predictions at each level were correct on the test set, for the active model
            ({layer.active_model}).
          </p>
          <table className="metrics">
            <thead>
              <tr>
                <th>Confidence level</th>
                <th>Estimated RTW probability</th>
                <th>Share of test patients</th>
                <th>Predictions correct</th>
              </tr>
            </thead>
            <tbody>
              {evaluation.confidence_bands.map((b, i, all) => {
                const lo = Math.round(b.min_distance * 100);
                const hi = all[i + 1] ? Math.round(all[i + 1].min_distance * 100) : null;
                const range = hi === null
                  ? `${50 + lo}% or more, or ${50 - lo}% or less`
                  : lo === 0
                    ? `between ${50 - hi}% and ${50 + hi}%`
                    : `${50 + lo}–${50 + hi}%, or ${50 - hi}–${50 - lo}%`;
                return (
                  <tr key={b.level}>
                    <td>{b.level}</td>
                    <td>{range}</td>
                    <td>{pct(b.share_of_test)}</td>
                    <td>{pct(b.accuracy)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {evaluation?.per_class && (
        <div className="panel">
          <h3>Accuracy by timing class</h3>
          <p className="sub">
            Active model ({layer.active_model}). Recall is the share of patients in each class
            the model identifies correctly.
          </p>
          <PerClassTable perClass={evaluation.per_class[layer.active_model]} labels={TIMING_ORDER} />
        </div>
      )}

      {evaluation && (
        <div className="panel">
          <h3>Most influential factors</h3>
          <p className="sub">
            Drop in ROC-AUC when each input field is shuffled in the test set, for the active
            model ({layer.active_model}). A larger drop means the model relies on that field more.
            These are statistical associations, not causes.
          </p>
          <FactorImportance factors={evaluation.factor_importance} />
        </div>
      )}

      <div className="panel">
        <h3>Feature importance</h3>
        <p className="sub">Top individual encoded features inside the active model ({layer.active_model}).</p>
        <FeatureImportance features={layer.top_features} />
      </div>
    </div>
  );
}
