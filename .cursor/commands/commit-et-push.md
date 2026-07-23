# Auto Commit

Quand je dis "commit" :
0. use rtk
1. Analyse les changements avec `git status` et avec `git diff -U0 -w --ignore-blank-lines| grep '^[+-]' | grep -v '^[+-][+-][+-]' | grep -v '^[+-][[:space:]]*$'`
2. Crée un message au format : `type: description courte`
3. Exécute : `git add -A && git commit -m "message" && git push`

Types : feat, fix, docs, style, refactor, test, chore

Messages en français, présent, max 50 caractères.
