# 📝 브라우저에 보여줄 HTML/CSS/JavaScript 화면 코드
INDEX_HTML = r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Leon's ChatBot</title>
  <style>
    :root {
      color-scheme: light;
      --surface: #f6f7fb;
      --panel: #ffffff;
      --ink: #1f2430;
      --muted: #6f768a;
      --line: #e0e4ee;
      --accent: #2e4780;
      --accent-soft: #eaf1fe;
      --danger: #c0392b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--surface);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    [hidden] { display: none !important; }
    button {
      border: 0;
      border-radius: 8px;
      padding: 11px 16px;
      background: var(--accent);
      color: #fff;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    button.secondary {
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid #cedffe;
    }
    button:disabled { opacity: .55; cursor: wait; }
    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      font: inherit;
      color: var(--ink);
      outline: none;
    }
    input:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-soft);
    }

    /* ── 로그인 화면 ── */
    #loginView {
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .login-card {
      width: min(400px, 100%);
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 36px 32px;
      box-shadow: 0 12px 32px rgba(31, 36, 48, .08);
      text-align: center;
    }
    .login-card h1 { margin: 0 0 4px; font-size: 26px; }
    .login-card .sub { margin: 0 0 24px; color: var(--muted); font-size: 14px; }
    .login-card form { display: grid; gap: 12px; text-align: left; }
    .login-card label { font-size: 13px; font-weight: 650; color: var(--muted); }
    .auth-error { min-height: 20px; color: var(--danger); font-size: 13px; margin: 4px 0 0; }
    .auth-toggle { margin-top: 16px; font-size: 14px; color: var(--muted); }
    .auth-toggle a { color: var(--accent); cursor: pointer; font-weight: 650; text-decoration: none; }

    /* ── 채팅 화면 ── */
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      padding: 14px 20px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }
    header h1 {
      grid-column: 2;         /* 📝 3열 그리드의 정가운데 칸에 배치 → 항상 화면 정중앙 */
      margin: 0;
      font-size: 20px;
      text-align: center;
    }
    .header-right {
      grid-column: 3;
      display: flex;
      gap: 8px;
      align-items: center;
      justify-content: flex-end;
    }
    .header-right .who { color: var(--muted); font-size: 14px; }
    .header-right button { padding: 8px 12px; font-size: 13px; }

    .chat-wrap {
      width: min(1080px, calc(100% - 32px));
      margin: 0 auto;
      padding: 24px 0 190px;
    }
    .turn { margin-bottom: 28px; }
    .question-row { display: flex; justify-content: flex-end; margin-bottom: 12px; }
    .question {
      max-width: min(75%, 640px);
      background: var(--accent);
      color: #fff;
      padding: 11px 15px;
      border-radius: 12px 12px 2px 12px;
      line-height: 1.55;
      white-space: pre-wrap;
    }
    .answers { display: grid; gap: 14px; }
    .answers[data-count="1"] { grid-template-columns: 1fr; }
    .answer-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px 16px;
      line-height: 1.65;
      white-space: pre-wrap;
    }
    .answer-card .badge {
      display: inline-block;
      font-size: 12px;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 999px;
      margin-bottom: 10px;
    }
    .answer-card.assistant { border-top: 3px solid var(--accent); }
    .answer-card.assistant .badge { background: var(--accent-soft); color: var(--accent); }
    .answer-card.local { border-top: 3px solid #2f9e63; }
    .answer-card.local .badge { background: #eaf7f0; color: #2f9e63; }
    .answer-card .error { color: var(--danger); font-size: 13px; }
    .thinking { color: var(--muted); font-style: italic; }

    .composer {
      position: fixed;
      bottom: 0;
      left: 0;
      right: 0;
      background: linear-gradient(transparent, var(--surface) 25%);
      padding: 16px 16px 20px;
    }
    .composer-box { width: min(1080px, 100%); margin: 0 auto; }
    .mode-toggle {
      display: inline-flex;
      gap: 4px;
      padding: 4px;
      margin-bottom: 10px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 4px 16px rgba(31, 36, 48, .06);
    }
    .mode-toggle button {
      min-width: 124px;
      padding: 8px 12px;
      background: transparent;
      color: var(--muted);
      border-radius: 6px;
      font-size: 13px;
    }
    .mode-toggle button.active {
      background: var(--accent);
      color: #fff;
    }
    .composer-inner { display: flex; gap: 10px; align-items: flex-end; }
    .composer textarea {
      min-height: 52px;
      max-height: 160px;
      resize: none;
      background: var(--panel);
      box-shadow: 0 4px 16px rgba(31, 36, 48, .08);
    }
    .empty-hint { text-align: center; color: var(--muted); margin-top: 80px; line-height: 1.8; }
    @media (max-width: 960px) {
      header h1 { font-size: 17px; }
      .header-right .who { display: none; }
    }
  </style>
</head>
<body>
  <!-- 📝 로그인 화면 -->
  <div id="loginView">
    <div class="login-card">
      <h1>Leon's ChatBot</h1>
      <p class="sub">로그인하고 바로 질문하세요</p>
      <form id="authForm">
        <label for="username">아이디</label>
        <input id="username" autocomplete="username" required minlength="2" maxlength="30">
        <label for="password">비밀번호</label>
        <input id="password" type="password" autocomplete="current-password" required minlength="6" maxlength="100">
        <button id="authSubmit" type="submit">로그인</button>
      </form>
      <p id="authError" class="auth-error"></p>
      <p class="auth-toggle">
        <span id="toggleText">계정이 없나요?</span>
        <a id="toggleLink">회원가입</a>
      </p>
    </div>
  </div>

  <!-- 📝 채팅 화면 -->
  <div id="chatView" hidden>
    <header>
      <h1>Leon's ChatBot</h1>
      <div class="header-right">
        <span class="who" id="whoami"></span>
        <button id="clearButton" class="secondary" type="button">기록 지우기</button>
        <button id="logoutButton" class="secondary" type="button">로그아웃</button>
      </div>
    </header>
    <main class="chat-wrap">
      <div id="chatLog"></div>
      <div id="emptyHint" class="empty-hint">
        질문을 입력하면 한 개의 답변만 표시됩니다.
      </div>
    </main>
    <div class="composer">
      <div class="composer-box">
        <div class="mode-toggle" role="group" aria-label="답변 모드">
          <button id="aiModeButton" class="active" type="button" data-responder="ai">AI가 답변</button>
          <button id="localModeButton" type="button" data-responder="local">로컬 LLM이 답변</button>
        </div>
        <form class="composer-inner" id="chatForm">
          <textarea id="chatInput" placeholder="질문을 입력하세요 (Enter로 전송, Shift+Enter 줄바꿈)" required></textarea>
          <button id="sendButton" type="submit">보내기</button>
        </form>
      </div>
    </div>
  </div>

  <script>
    // ── 상태 ──
    // 📝 토큰은 localStorage에 저장해서 새로고침해도 로그인이 유지되게 함
    let token = localStorage.getItem("lc_token") || "";
    let authMode = "login";  // login | signup
    let responderMode = localStorage.getItem("lc_responder") || "ai";

    const $ = (sel) => document.querySelector(sel);
    const loginView = $("#loginView"), chatView = $("#chatView");
    const authForm = $("#authForm"), authError = $("#authError");
    const authSubmit = $("#authSubmit"), toggleLink = $("#toggleLink"), toggleText = $("#toggleText");
    const chatLog = $("#chatLog"), emptyHint = $("#emptyHint");
    const chatForm = $("#chatForm"), chatInput = $("#chatInput"), sendButton = $("#sendButton");
    const modeButtons = document.querySelectorAll(".mode-toggle button");

    // ── API 호출 도우미 ──
    async function api(path, options = {}) {
      const response = await fetch(path, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          ...(token ? { "Authorization": `Bearer ${token}` } : {}),
        },
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "요청에 실패했습니다.");
      return data;
    }

    // ── 로그인 / 회원가입 ──
    function setAuthMode(next) {
      authMode = next;
      authSubmit.textContent = authMode === "login" ? "로그인" : "회원가입";
      toggleText.textContent = authMode === "login" ? "계정이 없나요?" : "이미 계정이 있나요?";
      toggleLink.textContent = authMode === "login" ? "회원가입" : "로그인";
      authError.textContent = "";
    }
    toggleLink.addEventListener("click", () => setAuthMode(authMode === "login" ? "signup" : "login"));

    authForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      authSubmit.disabled = true;
      authError.textContent = "";
      try {
        const data = await api(`/api/auth/${authMode === "login" ? "login" : "signup"}`, {
          method: "POST",
          body: JSON.stringify({
            username: $("#username").value.trim(),
            password: $("#password").value,
          }),
        });
        token = data.token;
        localStorage.setItem("lc_token", token);
        enterChat(data.username);
      } catch (error) {
        authError.textContent = error.message;
      } finally {
        authSubmit.disabled = false;
      }
    });

    // ── 화면 전환 ──
    async function enterChat(username) {
      loginView.hidden = true;
      chatView.hidden = false;
      $("#whoami").textContent = `${username} 님`;
      chatLog.textContent = "";
      // 📝 서버가 {question, answer, error} 모양으로 주므로 그대로 그림
      const turns = await api("/api/chat/history").catch(() => []);
      for (const turn of turns) renderTurn(turn.question, turn);
      emptyHint.hidden = turns.length > 0;
      window.scrollTo(0, document.body.scrollHeight);
      chatInput.focus();
    }

    function leaveChat() {
      token = "";
      localStorage.removeItem("lc_token");
      chatView.hidden = true;
      loginView.hidden = false;
      $("#password").value = "";
      setAuthMode("login");
    }

    $("#logoutButton").addEventListener("click", async () => {
      await api("/api/auth/logout", { method: "POST" }).catch(() => {});
      leaveChat();
    });

    $("#clearButton").addEventListener("click", async () => {
      if (!confirm("대화 기록을 모두 지울까요?")) return;
      await api("/api/chat/history", { method: "DELETE" }).catch(() => {});
      chatLog.textContent = "";
      emptyHint.hidden = false;
    });

    function setResponderMode(next) {
      responderMode = next === "local" ? "local" : "ai";
      localStorage.setItem("lc_responder", responderMode);
      modeButtons.forEach((button) => {
        const active = button.dataset.responder === responderMode;
        button.classList.toggle("active", active);
        button.setAttribute("aria-pressed", active ? "true" : "false");
      });
    }

    modeButtons.forEach((button) => {
      button.addEventListener("click", () => setResponderMode(button.dataset.responder));
    });
    setResponderMode(responderMode);

    // ── 채팅 렌더링 ──
    function answerCard(result) {
      const card = document.createElement("div");
      const responder = result.responder === "local" ? "local" : "assistant";
      card.className = `answer-card ${responder}`;
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = result.responder === "local" ? "로컬 LLM" : "AI";
      card.appendChild(badge);
      const body = document.createElement("div");
      if (result.pending) {
        body.className = "thinking";
        body.textContent = "답변 생성 중...";
      } else if (result.error) {
        body.className = "error";
        body.textContent = `오류: ${result.error}`;
      } else {
        body.textContent = result.answer || "(빈 응답)";
      }
      card.appendChild(body);
      return card;
    }

    function renderAnswer(container, result) {
      container.textContent = "";
      container.dataset.count = "1";
      container.appendChild(answerCard(result || { error: "응답이 없습니다." }));
    }

    function renderTurn(question, result) {
      const turn = document.createElement("div");
      turn.className = "turn";
      const questionRow = document.createElement("div");
      questionRow.className = "question-row";
      const bubble = document.createElement("div");
      bubble.className = "question";
      bubble.textContent = question;
      questionRow.appendChild(bubble);
      turn.appendChild(questionRow);
      const answers = document.createElement("div");
      answers.className = "answers";
      renderAnswer(answers, result);
      turn.appendChild(answers);
      chatLog.appendChild(turn);
      window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
      return turn;
    }

    function pendingAnswer() {
      return { responder: responderMode, pending: true };
    }

    // ── 질문 전송 ──
    chatForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = chatInput.value.trim();
      if (!message) return;
      chatInput.value = "";
      emptyHint.hidden = true;
      sendButton.disabled = true;
      const turn = renderTurn(message, pendingAnswer());
      try {
        const data = await api("/api/chat/ask", {
          method: "POST",
          body: JSON.stringify({ message, responder: responderMode }),
        });
        renderAnswer(turn.querySelector(".answers"), data);
      } catch (error) {
        renderAnswer(turn.querySelector(".answers"), { error: error.message });
        if (error.message.includes("로그인")) leaveChat();
      } finally {
        sendButton.disabled = false;
        chatInput.focus();
        window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
      }
    });

    chatInput.addEventListener("keydown", (event) => {
      // 📝 한글 IME 조합 중 Enter는 무시 → 마지막 글자가 한 번 더 전송되는 버그 방지
      if (event.isComposing || event.keyCode === 229) return;
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
      }
    });

    // ── 시작: 저장된 토큰이 있으면 자동 로그인 ──
    (async () => {
      if (!token) return;
      try {
        const data = await api("/api/auth/me");
        enterChat(data.username);
      } catch {
        leaveChat();
      }
    })();
  </script>
</body>
</html>
"""
