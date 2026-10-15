# Usability test kit

Everything needed to run the prototype testing planned in proposal §3.6.2:
can users enter patient information, move between the sections, produce
an RTW prediction, see the estimated timing, and understand the factors
shown? The result goes in the report as the evidence for the "Usability"
requirement (Table 3.8), next to the automated tests.

Not yet run. Planned for the final weeks of the project.

## 1. Before the sessions

- **Ask your supervisor** whether testing with people needs a consent form
  or approval at faculty level, and use the university's own form if there
  is one. A short consent statement is in §7 in case none is required.
- **Participants: 5 to 10.** Five people find most of the serious
  problems. Try to include at least one person who works in health care
  (doctor, physiotherapist, occupational therapist, nurse, or a student of
  one of these), since the system is meant for them. Record each person's
  background in one line, never their name.
- **Set up a fresh demo system on your own laptop**, not the hosted copy,
  which sleeps and loses its data:

  ```bash
  python api/seed_demo_data.py
  cd api && python -m uvicorn main:app --port 8000
  cd frontend && npm run dev
  ```

  The seed adds 12 invented patients and the demo accounts below (password
  for all: `demo12345`). If the database already holds the demo patients,
  stop the API, delete `api/sci_rtw.db` and seed again, so every
  participant starts from the same state.

  | Account | Role |
  |---|---|
  | `dr.aisyah` | Doctor |
  | `patient.farid` | Patient (linked to profile MRN-1001) |

- Print §3 (tasks and case cards) for the participant, and §4 and §5 for
  yourself, one copy per participant.

## 2. Running a session (about 20 minutes)

1. Read out: "I am testing the system, not you. Please think aloud as you
   go. I can't help with the tasks, but there are no wrong answers."
2. Hand over the task sheet. Time each task from when they start reading
   it to when they say they are done.
3. Do not help unless they are completely stuck for about two minutes. If
   you do help, mark the task "with help".
4. After the tasks, ask the comprehension questions (§5), then hand over
   the SUS questionnaire (§6).
5. Ask: "What was the most confusing part?" and "What would you change?"
   Write the answers down word for word.

## 3. Participant task sheet

*You are a doctor at a rehabilitation unit. Sign in with username
`dr.aisyah` and password `demo12345`.*

1. Create a new patient profile: Patient ID **TEST-01**, name **Test
   Patient One**.
2. For TEST-01, enter the assessment on **Case card A** and run the
   prediction.
3. Using the result on screen, answer:
   a. How likely is this patient to return to work?
   b. How sure is the system about this estimate?
   c. If they return, when is that most likely?
   d. Which entered detail raised the estimate the most?
4. Start a new assessment for a new patient, **TEST-02**, using **Case
   card B**, and run it.
5. Find the saved prediction for patient **MRN-1003** in the history,
   open it, and add the note "Discussed with family".
6. Print the report for that prediction (stop at the print preview).
7. Find out how accurate the system's predictions are overall.
8. Sign out, then sign in as the patient: username `patient.farid`,
   password `demo12345`. Find your own latest result.

**Case card A**

| Field | Value |
|---|---|
| Sex | Male |
| Age at injury | 18 |
| Highest education | High School Diploma or G.E.D. |
| Marital status | Never Married (Single) |
| Pre-injury employment | Working (competitive labor market) |
| Cause of injury | Auto accident |
| Neurologic level | Lumbar L03 |
| AIS grade | D |
| Injury completeness | Paraplegia, incomplete |
| ASIA motor score | 92 |

**Case card B**

| Field | Value |
|---|---|
| Sex | Female |
| Age at injury | 28 |
| Highest education | High School Diploma or G.E.D. |
| Marital status | Divorced |
| Pre-injury employment | Working (competitive labor market) |
| Cause of injury | Auto accident |
| Neurologic level | Cervical C03 |
| AIS grade | B |
| Injury completeness | Tetraplegia, incomplete |
| ASIA motor score | not assessed (leave blank) |

## 4. Observer sheet (one per participant)

Participant code: ____  Background: ______________________  Date: ______

| Task | Completed (yes / with help / no) | Time (mm:ss) | Errors or hesitations seen |
|---|---|---|---|
| 1 Create profile | | | |
| 2 Enter case A, predict | | | |
| 3a–d Read the result | | | |
| 4 Case B, predict | | | |
| 5 Find record, add note | | | |
| 6 Print report | | | |
| 7 Find model accuracy | | | |
| 8 Patient's own result | | | |

## 5. Comprehension questions (ask aloud after the tasks)

These test the part of §3.6.2 that says users should *understand* the
result, which task completion alone does not show. Mark each right or
wrong; the right answers are in brackets.

1. Case B came out as unlikely to return to work. Does that mean she will
   not return? *(No: it is an estimate based on similar past patients.)*
2. What does the confidence level tell you? *(How often the model was
   right for past patients whose estimate was as far from 50%.)*
3. Is the "most likely timing" as reliable as the yes/no estimate? *(No,
   the results page says timing is less reliable.)*
4. Do the factors shown mean, for example, that education *causes* return
   to work? *(No, they are statistical patterns in the data.)*
5. Should this result decide the patient's rehabilitation plan? *(No, it
   supports the clinician's judgement.)*

## 6. System Usability Scale (SUS)

A standard ten-item questionnaire (Brooke, 1996), free to use. Each item
is answered from 1 (strongly disagree) to 5 (strongly agree).

1. I think that I would like to use this system frequently.
2. I found the system unnecessarily complex.
3. I thought the system was easy to use.
4. I think that I would need the support of a technical person to be able
   to use this system.
5. I found the various functions in this system were well integrated.
6. I thought there was too much inconsistency in this system.
7. I would imagine that most people would learn to use this system very
   quickly.
8. I found the system very cumbersome to use.
9. I felt very confident using the system.
10. I needed to learn a lot of things before I could get going with this
    system.

**Scoring.** For items 1, 3, 5, 7 and 9, take the answer minus 1. For
items 2, 4, 6, 8 and 10, take 5 minus the answer. Add the ten numbers and
multiply by 2.5. The score runs from 0 to 100. It is not a percentage.

Example: answers 4, 2, 4, 1, 4, 2, 5, 1, 4, 2 give
(3+3+3+4+3+3+4+4+3+3) × 2.5 = 82.5.

**Reading it.** About 68 is the average across many published studies, so
a mean above 68 can be reported as above average. With 5 to 10 people,
report the mean together with the lowest and highest score, not the mean
alone.

## 7. Consent statement (only if the university has no form of its own)

> I agree to take part in a usability test of SCI-RTW Predict, a final
> year project prototype. I will use the system with invented patient
> data only. My name will not be recorded; my answers will be used only
> for this project's report. I may stop at any time.
>
> Signature: ____________  Date: ________

## 8. What to put in the report

| Measure | How to report it |
|---|---|
| Task completion | For each task, the share of participants who completed it without help |
| Time on task | Median time per task |
| Comprehension | For each question, the number answering correctly |
| SUS | Mean score with the range, compared with the benchmark of 68 |
| Problems found | Each problem, how many participants hit it, and what was changed |

If something is fixed between sessions, say which participants saw which
version.
