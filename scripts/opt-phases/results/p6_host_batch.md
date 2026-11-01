| Item | Old | New | Ratio | Hash-rate note |
|------|-----|-----|-------|----------------|
| Host batching | prestate, k-segments, pipeline already in `engine-cuda` | no further host change | — | CPU midstate already uses `custom::permute_mut`. GPU batch shape unchanged. |
