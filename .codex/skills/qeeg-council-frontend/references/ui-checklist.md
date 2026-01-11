# UI checklist

## Patients
- list view with search
- create patient modal
- patient detail page:
  - notes editor
  - reports list
  - run history list

## Reports
- upload (pdf or text)
- extracted text preview
- store returned report_id

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
