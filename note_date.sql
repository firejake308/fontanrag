select
  FILE_NAME,
  MRN,
  CASE
    WHEN sign_time IS NOT NULL THEN sign_time
    WHEN addend_time IS NOT NULL THEN addend_time
    WHEN note_time IS NOT NULL THEN note_time
    ELSE NULL
  END AS note_date
from
  notes
order by
  FILE_NAME