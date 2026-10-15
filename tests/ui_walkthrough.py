"""
Drives the real React UI in headless Chromium through the full demo flow
(register -> admin approval -> login -> patient profile -> prediction ->
history -> patient's own view) and saves screenshots.

Requires the API (port 8000) and frontend (port 5173) running.
    pip install playwright && playwright install chromium
    python tests/ui_walkthrough.py

Environment variables:
    WALK_URL             frontend address (default: http://localhost:5173)
    SHOTS                where screenshots go (default: docs/screenshots)
    WALK_ADMIN_PASSWORD  administrator password (default: admin123)
    WALK_ADMIN_NEW_PASSWORD  what the administrator's password is changed to
                         when the system asks for the default to be replaced,
                         and what is tried if WALK_ADMIN_PASSWORD is refused
                         (default: walkthrough-admin1)

Every run creates fresh accounts and patient profiles with a timestamp
suffix, so it can be re-run against the same database.
"""
import os
import re
import sys
import time
from playwright.sync_api import sync_playwright

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.environ.get("SHOTS", os.path.join(ROOT, "docs", "screenshots"))
os.makedirs(OUT, exist_ok=True)
URL = os.environ.get("WALK_URL", "http://localhost:5173")
ADMIN_PASSWORD = os.environ.get("WALK_ADMIN_PASSWORD", "admin123")
ADMIN_NEW_PASSWORD = os.environ.get("WALK_ADMIN_NEW_PASSWORD", "walkthrough-admin1")
RUN = str(int(time.time()))[-6:]
DOCTOR = f"dr_walk_{RUN}"
PATIENT_USER = f"pt_walk_{RUN}"
PASSWORD = "walkthrough-pw1"
errors = []


def field(page, label):
    return page.locator(".field", has_text=label)


def pick(page, label, option):
    field(page, label).locator("select").select_option(label=option)


def selected_text(page, label):
    return field(page, label).locator("select").evaluate("s => s.options[s.selectedIndex].text")


def sign_in_admin(page):
    """Sign in as the administrator, replacing the default password if the
    system asks for that (first run against a new database). On a later run
    the default no longer works, so the replacement is tried instead."""
    expected_statuses.add(401)
    with page.expect_response("**/auth/login") as login:
        sign_in(page, "admin", ADMIN_PASSWORD)
    if login.value.status != 200:
        with page.expect_response("**/auth/login"):
            sign_in(page, "admin", ADMIN_NEW_PASSWORD)
    expected_statuses.discard(401)
    page.wait_for_selector("h2:has-text('User Accounts'), h1:has-text('Choose a new password')", timeout=15000)
    if page.locator("h1:has-text('Choose a new password')").count():
        check("default administrator password must be changed before anything else",
              page.locator(".sidebar").count() == 0)
        field(page, "Current password").locator("input").fill(ADMIN_PASSWORD)
        page.fill("#new_password", ADMIN_NEW_PASSWORD)
        page.fill("#repeat_password", ADMIN_NEW_PASSWORD)
        page.click("button[type=submit]")


def settle(page):
    """Let hover/active colour transitions finish so screenshots are not caught mid-fade."""
    page.mouse.move(700, 5)
    page.wait_for_timeout(250)


def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        errors.append(name)


def nav_labels(page):
    return page.locator(".sidebar nav button").all_inner_texts()


def sign_in(page, username, password):
    page.click("button.model-tab:has-text('Sign in')")
    field(page, "Username").locator("input").fill(username)
    field(page, "Password").locator("input").fill(password)
    page.click("button[type=submit]")


def register(page, full_name, username, role_label):
    page.click("button.model-tab:has-text('Register')")
    field(page, "Full name").locator("input").fill(full_name)
    field(page, "Username").locator("input").fill(username)
    field(page, "Password").locator("input").fill(PASSWORD)
    pick(page, "I am a", role_label)
    page.click("button[type=submit]")


def log_out(page):
    page.click("button.logout")
    page.wait_for_selector(".auth-card")


# Same two real, model-verified fixtures used in tests/test_api.py (see that
# file's header comment for why hand-picked "plausible" patients are unsafe
# to use as test fixtures here).
RTW_LIKELY = dict(sex="Male", age=18, edu="High School Diploma or G.E.D.", marital="Never Married (Single)",
                   employment="Working (competitive labor market)", cause="Auto accident",
                   neuro="Lumbar L03", ais="D — Incomplete (motor preserved, most muscles ≥3)",
                   complete="Paraplegia, incomplete", motor=92)
RTW_UNLIKELY = dict(sex="Male", age=26, edu="8th grade or less", marital="Never Married (Single)",
                     employment="Working (competitive labor market)", cause="Auto accident",
                     neuro="Cervical C04", ais="D — Incomplete (motor preserved, most muscles ≥3)",
                     complete="Tetraplegia, incomplete", motor=46)


def create_profile_inline(page, code, name):
    field(page, "Patient ID").locator("input").fill(code)
    field(page, "Patient name").locator("input").fill(name)
    page.click("button:has-text('Create profile')")
    page.wait_for_function(
        "code => { const s = document.querySelector('#select_patient');"
        " return s && s.options[s.selectedIndex].text.includes(code); }", arg=code)


def fill_patient(page, p):
    pick(page, "Biological sex", p["sex"])
    field(page, "Age at injury").locator("input").fill(str(p["age"]))
    pick(page, "Highest education level", p["edu"])
    pick(page, "Marital status", p["marital"])
    pick(page, "Pre-injury employment status", p["employment"])
    page.click("button:has-text('Next')")
    pick(page, "Cause of injury", p["cause"])
    pick(page, "Neurologic level", p["neuro"])
    pick(page, "AIS grade", p["ais"])
    pick(page, "Injury completeness", p["complete"])
    page.click("button:has-text('Next')")
    field(page, "ASIA motor total").locator("input").fill(str(p["motor"]))
    page.click("button:has-text('Run Prediction')")
    page.wait_for_selector("text=Prediction Results", timeout=15000)
    page.wait_for_timeout(300)


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    problems = []
    expected_statuses = set()  # auth failures this script triggers on purpose
    page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    page.on("console", lambda m: problems.append(f"console: {m.text}")
            if m.type == "error" and "Failed to load resource" not in m.text else None)
    page.on("response", lambda r: problems.append(f"HTTP {r.status} {r.url}")
            if r.status >= 400 and "fonts.g" not in r.url and r.status not in expected_statuses else None)
    page.on("dialog", lambda d: d.accept())

    # ================= TC01: login + role-based access =================
    page.goto(URL)
    page.wait_for_selector(".auth-card")
    check("TC01 app opens on the sign-in screen, not the dashboard", page.locator(".sidebar").count() == 0)
    settle(page)
    page.screenshot(path=f"{OUT}/01_login.png")

    expected_statuses.update({401, 403})
    sign_in(page, "admin", "definitely-wrong-password")
    page.wait_for_selector(".error-banner")
    check("TC01 wrong password is rejected with a message",
          "Incorrect username or password" in page.inner_text(".error-banner"))

    register(page, "Dr Walk Through", DOCTOR, "Doctor")
    page.wait_for_selector("text=An administrator must approve it")
    check("TC01 clinician registration waits for administrator approval", True)
    settle(page)
    page.screenshot(path=f"{OUT}/02_register_pending.png")
    sign_in(page, DOCTOR, PASSWORD)
    page.wait_for_selector(".error-banner")
    check("TC01 unapproved clinician cannot sign in",
          "awaiting administrator approval" in page.inner_text(".error-banner"))
    expected_statuses.clear()

    # --- Administrator: manage user accounts ---
    sign_in_admin(page)
    page.wait_for_selector("h2:has-text('User Accounts')", timeout=15000)
    check("TC01 administrator lands on User Accounts", True)
    check("TC01 administrator menu has no patient functions",
          nav_labels(page) == ["User Accounts", "Model Insights"])
    row = page.locator("table.history tbody tr", has_text=DOCTOR)
    check("admin sees the pending registration as Inactive", "Inactive" in row.inner_text())
    settle(page)
    page.screenshot(path=f"{OUT}/03_admin_users_pending.png")
    row.locator("button:has-text('Approve')").click()
    page.wait_for_function(
        "u => [...document.querySelectorAll('table.history tbody tr')]"
        ".some(r => r.innerText.includes(u) && r.innerText.includes('Active') && !r.innerText.includes('Inactive'))",
        arg=DOCTOR)
    check("admin approves the account", True)
    page.click("button:has-text('Model Insights')")
    page.wait_for_selector("table.metrics")
    check("TC07 administrator can view the model accuracy report",
          page.locator(".panel", has_text="Model comparison").locator("table.metrics tbody tr").count() == 3)
    log_out(page)

    # --- Patient self-registration ---
    register(page, "Pat Walk Through", PATIENT_USER, "Patient")
    page.wait_for_selector("h2:has-text('My Results')", timeout=15000)
    check("TC01 patient account is usable immediately", True)
    check("TC01 patient menu is limited to My Results", nav_labels(page) == ["My Results"])
    check("patient with no linked record sees an empty state",
          "link your account" in page.inner_text(".empty-state"))
    settle(page)
    page.screenshot(path=f"{OUT}/04_patient_empty.png")
    log_out(page)

    # ================= Doctor: full clinical flow =================
    sign_in(page, DOCTOR, PASSWORD)
    page.wait_for_selector("text=Model API connected", timeout=15000)
    check("sidebar shows API connected", True)
    check("TC01 doctor menu has the clinical functions",
          nav_labels(page) == ["Dashboard", "Patients", "New Prediction", "Model Insights", "Patient History"])
    page.wait_for_selector(".stat-card .value:has-text('%')", timeout=10000)
    settle(page)
    page.screenshot(path=f"{OUT}/05_dashboard.png")

    # --- form: patient step + the 3 data-entry steps ---
    page.click(".sidebar button:has-text('New Prediction')")
    page.wait_for_selector("text=Patient profile")
    check("form has 4 steps (Patient + 3 data-entry)", page.locator(".step-tab").count() == 4)
    check("no Psychosocial step", page.locator(".step-tab", has_text="Psychosocial").count() == 0)
    next_btn = page.locator("button.btn.primary", has_text="Next")
    check("TC03 Next disabled until a patient profile is chosen", next_btn.is_disabled())
    code_a, code_b = f"MRN-A{RUN}", f"MRN-B{RUN}"
    create_profile_inline(page, code_a, "Alex Walkthrough")
    check("FR2 new patient profile is created and selected", code_a in selected_text(page, "Patient profile"))
    settle(page)
    page.screenshot(path=f"{OUT}/06_form_patient.png")
    next_btn.click()

    check("TC03 Next disabled while required fields empty", next_btn.is_disabled())
    check("U.S. veteran status is not asked (not applicable outside the U.S.)",
          page.locator(".field label", has_text="Veteran").count() == 0)
    pick(page, "Biological sex", "Male")
    field(page, "Age at injury").locator("input").fill("18")
    pick(page, "Highest education level", "High School Diploma or G.E.D.")
    pick(page, "Marital status", "Never Married (Single)")
    pick(page, "Pre-injury employment status", "Working (competitive labor market)")
    settle(page)
    page.screenshot(path=f"{OUT}/07_form_demographics.png")
    check("TC02 Next enabled once required fields filled", next_btn.is_enabled())
    next_btn.click()
    check("Vertebral injury defaults to 'Not specified'",
          selected_text(page, "Vertebral fracture") == "Not specified")
    pick(page, "Cause of injury", RTW_LIKELY["cause"])
    pick(page, "Neurologic level", RTW_LIKELY["neuro"])
    pick(page, "AIS grade", RTW_LIKELY["ais"])
    pick(page, "Injury completeness", RTW_LIKELY["complete"])
    settle(page)
    page.screenshot(path=f"{OUT}/08_form_injury.png")
    page.click("button:has-text('Next')")
    check("Functional Status is the final step", page.locator(".step-tab.active").inner_text() == "Functional Status")
    field(page, "ASIA motor total").locator("input").fill(str(RTW_LIKELY["motor"]))
    settle(page)
    page.screenshot(path=f"{OUT}/09_form_functional.png")
    page.click("button:has-text('Run Prediction')")
    page.wait_for_selector("text=Prediction Results", timeout=15000)
    page.wait_for_timeout(300)
    settle(page)
    page.screenshot(path=f"{OUT}/10_results_rtw.png", full_page=True)

    body = page.inner_text("body")
    check("TC04 RTW case: verdict shown", "Likely to return to work" in body)
    check("results name the patient", code_a in body)
    check("TC05 RTW case: timing panel shown", "Estimated timing of first return to work" in body)
    check("RTW case: 'By Year 10' label shown (not old 'Year 10 or later')",
          "By Year 10" in body and "Year 10 or later" not in body)
    factor_text = page.locator(".panel", has_text="What drove this prediction").inner_text()
    check("factors show direction text", ("Raises RTW likelihood" in factor_text) or ("Lowers RTW likelihood" in factor_text))
    check("factors contain no raw snake_case/codes", re.search(r"[a-z]+_[a-z]+", factor_text) is None)
    check("no dropped fields (race/PHQ/job class) appear in factors", not re.search(r"(?i)race|phq|depression|job class", factor_text))
    badge = page.locator(".result-badge.rtw").inner_text()
    check("Fig 3.13 confidence level shown with its measured accuracy",
          re.search(r"Confidence level: (High|Moderate)", badge) is not None
          and re.search(r"correct \d+% of the time", badge) is not None)
    considerations = page.locator(".panel", has_text="Suggested considerations")
    check("Fig 3.13 suggested considerations shown, each with its reason",
          considerations.locator(".suggestion").count() >= 1
          and considerations.locator(".suggestion .why").count() == considerations.locator(".suggestion").count())
    check("RTW case leads with return-to-work planning",
          "Begin return-to-work planning early" in considerations.locator(".suggestion").first.inner_text())
    check("considerations are labelled as not being treatment recommendations",
          "not treatment recommendations" in considerations.inner_text())

    # --- Non-RTW path, second patient ---
    page.click("button:has-text('New Assessment')")
    page.wait_for_selector("text=Patient profile")
    create_profile_inline(page, code_b, "Blair Walkthrough")
    page.click("button:has-text('Next')")
    fill_patient(page, RTW_UNLIKELY)
    settle(page)
    page.screenshot(path=f"{OUT}/11_results_non_rtw.png", full_page=True)
    body = page.inner_text("body")
    check("TC04 Non-RTW case: verdict shown", "Unlikely to return to work" in body)
    check("TC05 Non-RTW case: no timing panel", "Estimated timing of first return to work" not in body)
    first = page.locator(".panel", has_text="Suggested considerations").locator(".suggestion").first.inner_text()
    check("Non-RTW case leads with a vocational rehabilitation referral prompt",
          "Consider early vocational rehabilitation referral" in first)

    # --- TC06: history is stored on the server ---
    page.click(".sidebar button:has-text('Patient History')")
    page.wait_for_selector("table.history")
    rows = page.locator("table.history tbody tr")
    check("TC06 history lists both saved predictions",
          rows.filter(has_text=code_a).count() == 1 and rows.filter(has_text=code_b).count() == 1)
    settle(page)
    page.screenshot(path=f"{OUT}/12_history.png")
    page.reload()
    page.wait_for_selector("text=Model API connected")
    check("TC01 session survives a page reload", page.locator(".sidebar").count() == 1)
    page.click(".sidebar button:has-text('Patient History')")
    page.wait_for_selector("table.history")
    check("TC06 history persists after reload",
          page.locator("table.history tbody tr").filter(has_text=code_a).count() == 1)
    page.evaluate("localStorage.removeItem('sci_rtw_prediction_history')")
    check("TC06 history does not depend on browser storage",
          page.evaluate("localStorage.getItem('sci_rtw_prediction_history')") is None
          and page.locator("table.history tbody tr").filter(has_text=code_b).count() == 1)

    # --- open a saved record + clinician notes (Prediction Report) ---
    page.locator("table.history tbody tr", has_text=code_a).locator("button:has-text('View')").click()
    page.wait_for_selector("text=Clinician notes")
    body = page.inner_text("body")
    check("saved record reopens with its factors", "What drove this prediction" in body)
    check("saved record keeps its confidence level and considerations",
          re.search(r"Confidence level: (High|Moderate)", body) is not None
          and "Begin return-to-work planning early" in body)
    note = "Refer to vocational rehabilitation; review at 6 months."
    page.locator("textarea[aria-label='Clinician notes']").fill(note)
    page.click("button:has-text('Save notes')")
    page.wait_for_selector("text=Notes saved.")
    check("clinician notes are saved", True)
    settle(page)
    page.screenshot(path=f"{OUT}/13_history_record_notes.png", full_page=True)

    # --- printed report: the same record as it comes out on paper ---
    check("results page offers a Print report button",
          page.locator("button:has-text('Print report')").count() == 1)
    page.emulate_media(media="print")
    page.set_viewport_size({"width": 794, "height": 1123})  # A4 at 96 dpi
    check("printed report leaves out the sidebar and buttons",
          not page.locator(".sidebar").is_visible()
          and not page.locator("button:has-text('Back to history')").is_visible())
    header = page.locator(".report-header")
    check("printed report is headed with the patient and who entered it",
          header.is_visible() and code_a in header.inner_text() and "Entered by" in header.inner_text())
    check("printed report shows the clinician notes as text",
          page.locator("p.notes-text", has_text=note).is_visible())
    page.screenshot(path=f"{OUT}/20_printed_report.png", full_page=True)
    page.emulate_media(media="screen")
    page.set_viewport_size({"width": 1400, "height": 900})
    page.click("button:has-text('Back to history')")

    # --- FR2: manage patient profiles; link the patient's own account ---
    page.click(".sidebar button:has-text('Patients')")
    page.wait_for_selector("h2:has-text('Patient Profiles')")
    row = page.locator("table.history tbody tr", has_text=code_a)
    check("FR2 profile list shows the assessment count", row.locator("td").nth(3).inner_text() == "1")
    row.locator("button:has-text('Edit')").click()
    field(page, "Linked patient account").locator("input").fill(PATIENT_USER)
    field(page, "Notes").locator("textarea").fill("Walkthrough profile.")
    page.click("button:has-text('Save profile')")
    page.wait_for_function(
        "u => [...document.querySelectorAll('table.history tbody tr')].some(r => r.innerText.includes(u))",
        arg=PATIENT_USER)
    check("FR2 profile updated and linked to the patient's account", True)
    settle(page)
    page.screenshot(path=f"{OUT}/14_patients.png")
    page.locator("input[aria-label='Search patients']").fill(code_b)
    check("FR2 patient search filters the list", page.locator("table.history tbody tr").count() == 1)

    # --- Model insights ---
    page.click(".sidebar button:has-text('Model Insights')")
    page.wait_for_selector("table.metrics")
    settle(page)
    page.screenshot(path=f"{OUT}/15_insights_l1.png", full_page=True)
    comparison = page.locator(".panel", has_text="Model comparison")
    rows = comparison.locator("table.metrics tbody tr")
    check("TC07 insights lists 3 models", rows.count() == 3)
    check("XGBoost (main) listed first", "XGBoost" in rows.nth(0).inner_text())
    check("TC07 majority-class reference point is shown", "always predicting" in comparison.inner_text())
    validation = page.locator(".panel", has_text="5-fold CV accuracy")
    check("TC07 validation table shows cross-validation and train-vs-test for 3 models",
          validation.locator("table.metrics tbody tr").count() == 3 and "±" in validation.inner_text())
    factors = page.locator(".panel", has_text="Most influential factors")
    check("Objective (a): factor ranking lists all 14 input fields", factors.locator(".factor-row").count() == 14)
    levels = page.locator(".panel", has_text="What the confidence level means").locator("table.metrics tbody tr")
    check("Model Insights explains the three confidence levels",
          [r.split("\t")[0] for r in levels.all_inner_texts()] == ["Low", "Moderate", "High"])
    check("factor ranking is readable (no raw names)", re.search(r"[a-z]+_[a-z]+", factors.inner_text()) is None)
    fi = page.locator(".panel", has_text="Feature importance").inner_text()
    check("feature importance readable (no raw names)", re.search(r"[a-z]+_[a-z]+", fi) is None)
    check("no dropped fields in feature importance", not re.search(r"(?i)\brace\b|phq|job class", fi))
    page.click("button:has-text('Layer 2')")
    page.wait_for_selector("text=Accuracy by timing class")
    per_class = page.locator(".panel", has_text="Accuracy by timing class").locator("table.metrics tbody tr")
    check("Layer 2 shows accuracy for each timing class in chronological order",
          [r.split("\t")[0] for r in per_class.all_inner_texts()] == ["Year 1", "Year 5", "By Year 10"])
    settle(page)
    page.screenshot(path=f"{OUT}/16_insights_l2.png", full_page=True)

    page.click(".sidebar button:has-text('Dashboard')")
    page.wait_for_selector("text=Recent assessments")
    check("dashboard shows recent assessments", code_b in page.locator(".panel", has_text="Recent assessments").inner_text())
    settle(page)
    page.screenshot(path=f"{OUT}/17_dashboard_with_history.png")
    log_out(page)
    check("TC01 logging out returns to the sign-in screen", page.locator(".sidebar").count() == 0)

    # ================= Patient: sees only their own results =================
    sign_in(page, PATIENT_USER, PASSWORD)
    page.wait_for_selector("table.history", timeout=15000)
    rows = page.locator("table.history tbody tr")
    check("TC06 patient sees exactly their own linked result", rows.count() == 1)
    check("patient does not see other patients' records", code_b not in page.inner_text("body"))
    settle(page)
    page.screenshot(path=f"{OUT}/18_patient_my_results.png")
    rows.first.locator("button:has-text('View')").click()
    page.wait_for_selector("text=Clinician notes")
    body = page.inner_text("body")
    check("patient can read the clinician's notes", note in body)
    check("patient cannot edit the notes", page.locator("textarea").count() == 0)
    settle(page)
    page.screenshot(path=f"{OUT}/19_patient_result_detail.png", full_page=True)

    check("no page errors / failed requests (excluding Google Fonts)", len(problems) == 0)
    if problems:
        print("problems:", problems)
    browser.close()

print(f"\nScreenshots saved to {OUT}")
print("FAILURES:", errors if errors else "none")
sys.exit(1 if errors else 0)
