# Как запушить на GitHub (если `git push` из Cursor не прошёл)

Репозиторий: https://github.com/samsa2846-sys/PEcf09  

Локально уже выполнены `git init`, первый коммит и ветка `main`. Осталось отправить код **с вашей машины**, где есть доступ к GitHub.

## Вариант A: PowerShell / CMD (HTTPS)

```powershell
cd c:\Cursor\DZ_2\rag-yandex-assistant
& "C:\Program Files\Git\cmd\git.exe" push -u origin main
```

Когда спросят логин/пароль: **логин** — ваш GitHub username, **пароль** — [Personal Access Token](https://github.com/settings/tokens) с правом `repo`.

## Вариант B: GitHub Desktop

File → Add local repository → выберите папку `rag-yandex-assistant` → Publish repository / Push.

## Важно

Файл `.env` в коммит **не попадает** (в `.gitignore`). После сброса шаблона снова скопируйте ключи из Yandex Cloud в локальный `.env`. Если ключ когда-либо светился в открытом виде — **создайте новый ключ** в консоли Yandex.
