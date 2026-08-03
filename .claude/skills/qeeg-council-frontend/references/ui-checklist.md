# UI checklist

## Patients
- list view with search
- create patient modal
- bulk upload page:
  - each file carries its patient's name and date of birth; the backend matches
    an existing chart or allocates the canonical clinic ID `XX_MM-DD-YYYY[_N]`
  - one patient per person, not one per file
  - a 409 name conflict is asked, not resolved by creating another patient
  - show created/skipped/errors summary and allow jumping to a created patient
- patient detail page:
  - notes editor
  - reports list
  - patient files section (PDF/Markdown/MP4 upload + list + delete)
  - run history list

## Reports
- upload (pdf or text)
- extracted text preview
- store returned report_id
- view extracted (opens /api/reports/{report_id}/extracted)
- re-extract (OCR) (POST /api/reports/{report_id}/reextract)
- view original PDF (GET /api/reports/{report_id}/original)
- view extracted page images (GET /api/reports/{report_id}/pages and /pages/{page_num})
- view extraction metadata when available (GET /api/reports/{report_id}/metadata)

## Runs
- run wizard:
  - pick report_id
  - pick council models from /api/models discovered list
  - pick consolidator
- run page:
  - stage progress (1..6)
  - artifact tabs
  - vote panel (stage 5)
  - final draft compare (stage 6)
  - selection action persisted to backend
  - export buttons
