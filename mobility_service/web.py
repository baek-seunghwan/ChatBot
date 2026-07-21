INDEX_HTML = r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MoveOps · 카카오 T 배송 관제</title>
  <style>
    :root {
      --yellow: #fee500;
      --yellow-dark: #e5cf00;
      --ink: #191919;
      --muted: #747474;
      --line: #e7e7e7;
      --surface: #f7f7f5;
      --panel: #fff;
      --success: #137a49;
      --danger: #c43c35;
      --blue: #315efb;
      --violet: #5347ff;
      --mint: #0f8a73;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "SUIT Variable", "Pretendard", "Noto Sans KR", "Apple SD Gothic Neo", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 6%, rgba(83, 71, 255, .11), transparent 24%),
        radial-gradient(circle at 88% 88%, rgba(15, 138, 115, .08), transparent 26%),
        var(--surface);
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 18px clamp(20px, 5vw, 72px);
      background: linear-gradient(110deg, #111111, #24203f 68%, #131313);
      color: white;
      border-bottom: 1px solid rgba(255, 255, 255, .08);
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-mark {
      display: grid;
      place-items: center;
      width: 40px;
      height: 40px;
      border-radius: 14px;
      background: var(--yellow);
      color: var(--ink);
      font-weight: 900;
    }
    .brand h1 { margin: 0; font-size: 19px; }
    .brand p { margin: 3px 0 0; color: #bdbdbd; font-size: 12px; }
    .status { font-size: 13px; color: #ddd; }
    .status b { color: var(--yellow); }
    .sandbox {
      padding: 10px 20px;
      background: #fff8cd;
      border-bottom: 1px solid #f0dd63;
      color: #594f0a;
      text-align: center;
      font-size: 13px;
    }
    main {
      width: min(1180px, calc(100% - 32px));
      margin: 28px auto 64px;
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(300px, .75fr);
      gap: 22px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 24px;
      box-shadow: 0 12px 32px rgba(20, 18, 36, .07);
    }
    .map-panel { grid-column: 1 / -1; padding: 0; overflow: hidden; }
    .map-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      padding: 22px 24px 18px;
    }
    .map-header .description { margin-bottom: 0; }
    .map-tools { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }
    .map-mode {
      padding: 10px 13px;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
    }
    .map-mode.active {
      border-color: var(--ink);
      background: var(--ink);
      color: white;
    }
    .map-mode.active[data-mode="pickup"] { background: var(--blue); border-color: var(--blue); }
    .map-mode.active[data-mode="dropoff"] { background: var(--danger); border-color: var(--danger); }
    .map-stage {
      position: relative;
      min-height: 420px;
      border-top: 1px solid var(--line);
      background:
        radial-gradient(circle at 30% 30%, rgba(254, 229, 0, .16), transparent 34%),
        #eef0ec;
    }
    #map { width: 100%; height: 420px; }
    .map-message {
      position: absolute;
      z-index: 4;
      inset: 0;
      display: grid;
      place-items: center;
      padding: 30px;
      color: var(--muted);
      text-align: center;
      background: rgba(247, 247, 245, .92);
    }
    .map-message.hidden { display: none; }
    .map-label {
      position: relative;
      border: 2px solid white;
      border-radius: 999px;
      padding: 7px 11px;
      box-shadow: 0 5px 16px rgba(0, 0, 0, .2);
      color: white;
      font-size: 12px;
      font-weight: 850;
      white-space: nowrap;
    }
    .map-label.pickup { background: var(--blue); }
    .map-label.dropoff { background: var(--danger); }
    .route-summary {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 12px 24px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }
    .route-summary b { color: var(--ink); }
    .route-dot { width: 8px; height: 8px; border-radius: 50%; }
    .route-dot.pickup { background: var(--blue); }
    .route-dot.dropoff { background: var(--danger); }
    h2 { margin: 0 0 6px; font-size: 21px; }
    .description { margin: 0 0 22px; color: var(--muted); font-size: 14px; }
    .section-title {
      margin: 24px 0 12px;
      padding-top: 20px;
      border-top: 1px solid var(--line);
      font-size: 15px;
    }
    .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 13px; }
    .grid.three { grid-template-columns: 1.25fr .75fr .75fr; }
    label { display: grid; gap: 6px; color: #4d4d4d; font-size: 12px; font-weight: 700; }
    .address-input { display: flex; gap: 7px; }
    .address-input input { min-width: 0; }
    .address-input button {
      flex: 0 0 auto;
      padding: 10px 12px;
      border: 1px solid #d8d8d8;
      background: #f3f3f3;
      color: var(--ink);
      font-size: 12px;
    }
    input, select {
      width: 100%;
      border: 1px solid #d8d8d8;
      border-radius: 10px;
      padding: 11px 12px;
      background: white;
      color: var(--ink);
      font: inherit;
      outline: none;
    }
    input:focus, select:focus {
      border-color: var(--ink);
      box-shadow: 0 0 0 3px rgba(25, 25, 25, .08);
    }
    .actions { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 22px; }
    .safety-note {
      margin-top: 14px;
      padding: 11px 12px;
      border-radius: 12px;
      border: 1px solid #f0dc74;
      background: #fff8d6;
      color: #5f4e08;
      font-size: 12px;
      line-height: 1.45;
      font-weight: 650;
    }
    button {
      border: 0;
      border-radius: 11px;
      padding: 12px 16px;
      background: linear-gradient(120deg, #191919, #2b2b37);
      color: white;
      font: inherit;
      font-size: 14px;
      font-weight: 750;
      cursor: pointer;
      transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
    }
    button:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(26, 23, 48, .14); }
    button.primary { background: linear-gradient(120deg, #ffe665, #fee500 66%); color: var(--ink); }
    button.secondary { background: linear-gradient(120deg, #f4f4ff, #eceff8); color: var(--ink); }
    button.danger { background: #fff0ef; color: var(--danger); }
    button:disabled { opacity: .48; cursor: wait; }
    .result {
      display: none;
      margin-top: 20px;
      border-radius: 18px;
      padding: 16px 16px 14px;
      background: linear-gradient(160deg, #f7f8fc, #f1f3f8);
      border: 1px solid #e2e6f0;
      overflow: auto;
    }
    .result.visible { display: block; }
    .result h3 { margin: 0 0 10px; font-size: 14px; letter-spacing: .01em; }
    .result-meta {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
      color: #5f6676;
      font-size: 12px;
    }
    .result-chip {
      border-radius: 999px;
      padding: 5px 9px;
      background: white;
      border: 1px solid #d7dceb;
      color: #3f4b67;
      font-size: 11px;
      font-weight: 700;
      white-space: nowrap;
    }
    .result-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .fare-card {
      border: 1px solid #dbe1ef;
      border-radius: 14px;
      padding: 12px;
      background: white;
      box-shadow: 0 4px 12px rgba(31, 39, 74, .05);
    }
    .fare-card.selected {
      border-color: #c8b938;
      box-shadow: 0 0 0 2px rgba(254, 229, 0, .45), 0 6px 16px rgba(31, 39, 74, .08);
    }
    .fare-title {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin: 0 0 8px;
      font-size: 13px;
      font-weight: 800;
      color: #263151;
    }
    .fare-sub {
      margin: 0 0 10px;
      color: #667089;
      font-size: 11px;
      font-weight: 700;
    }
    .fare-row {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-top: 5px;
      font-size: 12px;
    }
    .fare-label { color: #6f7788; }
    .fare-value { color: #17213d; font-weight: 800; }
    .fare-tag {
      display: inline-block;
      margin-bottom: 8px;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 10px;
      font-weight: 800;
      color: #5f4e08;
      background: #fff4b5;
      border: 1px solid #ebd56f;
    }
    pre {
      margin: 0;
      padding: 12px;
      border-radius: 12px;
      background: #fff;
      border: 1px solid #dfe4f1;
      font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: pre-wrap;
    }
    .result-body.hidden { display: none; }
    .order-list { display: grid; gap: 9px; margin-top: 16px; }
    .order {
      border: 1px solid var(--line);
      border-radius: 13px;
      padding: 13px;
      cursor: pointer;
      transition: .15s ease;
    }
    .order:hover { border-color: #aaa; transform: translateY(-1px); }
    .order-top { display: flex; justify-content: space-between; gap: 10px; }
    .order-id { overflow: hidden; text-overflow: ellipsis; font-size: 13px; font-weight: 750; }
    .badge {
      flex: 0 0 auto;
      border-radius: 999px;
      padding: 3px 8px;
      background: #ececec;
      font-size: 10px;
      font-weight: 800;
    }
    .order time { display: block; margin-top: 6px; color: var(--muted); font-size: 11px; }
    .tracking { display: flex; gap: 8px; margin-top: 14px; }
    .tracking button { flex: 0 0 auto; }
    .empty { color: var(--muted); font-size: 13px; text-align: center; padding: 26px 0; }
    .error { color: var(--danger); }
    footer { color: var(--muted); text-align: center; font-size: 12px; padding: 0 20px 36px; }
    .chat-fab {
      position: fixed;
      left: 24px;
      bottom: 24px;
      z-index: 60;
      width: 56px;
      height: 56px;
      border-radius: 50%;
      background: var(--yellow);
      color: var(--ink);
      font-size: 24px;
      display: grid;
      place-items: center;
      padding: 0;
      box-shadow: 0 10px 26px rgba(20, 18, 36, .22);
    }
    .chat-fab:hover { transform: translateY(-2px); box-shadow: 0 14px 30px rgba(20, 18, 36, .28); }
    .chat-widget {
      position: fixed;
      left: 24px;
      bottom: 92px;
      z-index: 60;
      width: 340px;
      max-height: 480px;
      display: flex;
      flex-direction: column;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 18px 42px rgba(20, 18, 36, .18);
      overflow: hidden;
    }
    .chat-widget.hidden { display: none; }
    .chat-widget-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 16px;
      background: linear-gradient(110deg, #111111, #24203f 68%, #131313);
      color: white;
    }
    .chat-widget-header h3 { margin: 0; font-size: 14px; }
    .chat-widget-header button {
      background: transparent;
      padding: 4px 8px;
      font-size: 16px;
      box-shadow: none;
    }
    .chat-widget-header button:hover { transform: none; box-shadow: none; opacity: .8; }
    .chat-log {
      flex: 1;
      overflow-y: auto;
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      background: var(--surface);
      min-height: 220px;
    }
    .chat-bubble {
      max-width: 85%;
      padding: 9px 12px;
      border-radius: 14px;
      font-size: 13px;
      line-height: 1.5;
      white-space: pre-wrap;
    }
    .chat-bubble.user {
      align-self: flex-end;
      background: var(--ink);
      color: white;
      border-bottom-right-radius: 4px;
    }
    .chat-bubble.bot {
      align-self: flex-start;
      background: white;
      border: 1px solid var(--line);
      color: var(--ink);
      border-bottom-left-radius: 4px;
    }
    .chat-widget-form {
      display: flex;
      gap: 8px;
      padding: 12px;
      border-top: 1px solid var(--line);
      background: white;
    }
    .chat-widget-form input { flex: 1; }
    .chat-widget-form button { flex: 0 0 auto; padding: 11px 14px; }
    .chat-mode-toggle {
      display: flex;
      gap: 6px;
      padding: 10px 12px 0;
      background: var(--surface);
    }
    .chat-mode-btn {
      flex: 1;
      padding: 7px 10px;
      font-size: 12px;
      font-weight: 700;
      border-radius: 999px;
      background: white;
      border: 1px solid var(--line);
      color: var(--muted);
      box-shadow: none;
    }
    .chat-mode-btn:hover { transform: none; box-shadow: none; }
    .chat-mode-btn.active { background: var(--ink); color: white; border-color: var(--ink); }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      .grid, .grid.three { grid-template-columns: 1fr; }
      .status { display: none; }
      .map-header { display: grid; }
      .map-tools { justify-content: flex-start; }
      #map, .map-stage { height: 340px; min-height: 340px; }
      .result-grid { grid-template-columns: 1fr; }
      .chat-widget { left: 16px; right: 16px; width: auto; bottom: 84px; }
      .chat-fab { left: 16px; bottom: 16px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-mark">M</div>
      <div><h1>MoveOps</h1><p>카카오 T 퀵·도보 배송 관제</p></div>
    </div>
    <div class="status" id="status">연동 상태 확인 중</div>
  </header>
  <div class="sandbox">Sandbox 환경입니다. 실제 개인정보 대신 테스트용 이름·전화번호를 사용하세요.</div>

  <main>
    <section class="panel map-panel">
      <div class="map-header">
        <div>
          <h2>배송 경로 지도</h2>
          <p class="description">주소를 검색하거나 지도를 클릭해 출발지와 도착지를 지정하세요.</p>
        </div>
        <div class="map-tools" aria-label="지도 좌표 선택">
          <button class="map-mode active" data-mode="pickup" onclick="setMapMode('pickup')">출발지 설정</button>
          <button class="map-mode" data-mode="dropoff" onclick="setMapMode('dropoff')">도착지 설정</button>
          <button class="map-mode" onclick="fitRoute()">경로 전체보기</button>
        </div>
      </div>
      <div class="map-stage">
        <div id="map" aria-label="카카오 배송 경로 지도"></div>
        <div class="map-message" id="mapMessage">카카오 지도를 불러오는 중입니다.</div>
      </div>
      <div class="route-summary">
        <span class="route-dot pickup"></span><b>출발지</b>
        <span>→</span>
        <span class="route-dot dropoff"></span><b>도착지</b>
        <span id="routeDistance">직선거리를 계산하는 중입니다.</span>
      </div>
    </section>

    <section class="panel">
      <h2>배송 요청</h2>
      <p class="description">출발지와 도착지를 입력해 예상 시간과 가격을 확인하세요.</p>

      <div class="grid">
        <label>배송 상품
          <select id="orderType">
            <option value="QUICK">퀵</option>
            <option value="QUICK_ECONOMY">퀵 이코노미</option>
            <option value="QUICK_EXPRESS">퀵 급송</option>
            <option value="DOBO">도보 배송</option>
          </select>
        </label>
        <label>물품 크기
          <select id="productSize">
            <option value="XS">XS · 서류/초소형</option>
            <option value="S">S · 소형</option>
            <option value="M">M · 중형</option>
            <option value="L">L · 대형</option>
          </select>
        </label>
      </div>

      <h3 class="section-title">출발지</h3>
      <div class="grid three">
        <label>주소
          <span class="address-input">
            <input id="pickupAddress" value="경기도 성남시 분당구 판교역로 152">
            <button type="button" onclick="searchAddress('pickup')">검색</button>
          </span>
        </label>
        <label>위도<input id="pickupLat" type="number" step="any" value="37.3946095"></label>
        <label>경도<input id="pickupLng" type="number" step="any" value="127.1118735"></label>
      </div>
      <div class="grid" style="margin-top:13px">
        <label>보내는 사람<input id="pickupName" value="테스트발송자"></label>
        <label>테스트 전화번호<input id="pickupPhone" value="010-1000-0001"></label>
      </div>

      <h3 class="section-title">도착지</h3>
      <div class="grid three">
        <label>주소
          <span class="address-input">
            <input id="dropoffAddress" value="경기도 성남시 분당구 정자동 49-4">
            <button type="button" onclick="searchAddress('dropoff')">검색</button>
          </span>
        </label>
        <label>위도<input id="dropoffLat" type="number" step="any" value="37.3595316"></label>
        <label>경도<input id="dropoffLng" type="number" step="any" value="127.1052133"></label>
      </div>
      <div class="grid" style="margin-top:13px">
        <label>받는 사람<input id="dropoffName" value="테스트수신자"></label>
        <label>테스트 전화번호<input id="dropoffPhone" value="010-1000-0002"></label>
      </div>

      <h3 class="section-title">물품 정보</h3>
      <div class="grid">
        <label>물품명<input id="productName" value="테스트 서류"></label>
        <label>신고 가격<input id="declaredValue" type="number" min="0" value="10000"></label>
      </div>

      <div class="actions">
        <button class="secondary" onclick="callEstimate(this)">예상시간 조회</button>
        <button class="secondary" onclick="callPrice(this)">가격 조회</button>
        <button class="primary" onclick="createOrder(this)">Sandbox 주문 접수</button>
      </div>
      <div class="safety-note">
        안전장치: 주문 접수 전 테스트 전화번호(010-1000-XXXX) 확인 + <b>SANDBOX</b> 입력 확인을 통과해야 요청이 전송됩니다.
      </div>
      <div class="result" id="result">
        <h3 id="resultTitle">결과</h3>
        <div class="result-meta" id="resultMeta"></div>
        <div class="result-grid" id="resultGrid"></div>
        <pre id="resultBody" class="result-body"></pre>
      </div>
    </section>

    <aside class="panel">
      <h2>주문 추적</h2>
      <p class="description">저장된 주문과 수신한 콜백 상태를 확인합니다.</p>
      <div class="tracking">
        <input id="trackingId" placeholder="partnerOrderId">
        <button onclick="trackOrder()">조회</button>
      </div>
      <div class="actions">
        <button class="secondary" onclick="refreshProvider()">카카오 상태 동기화</button>
        <button class="danger" onclick="cancelTrackedOrder()">주문 취소</button>
      </div>
      <div class="result" id="trackResult"><h3>주문 상세</h3><pre id="trackBody"></pre></div>

      <h3 class="section-title">최근 주문</h3>
      <div class="order-list" id="orderList"><div class="empty">주문을 불러오는 중입니다.</div></div>
    </aside>
  </main>
  <footer>MoveOps 개인 프로젝트 · 카카오 T 퀵·도보 배송 Sandbox 연동</footer>

  <button class="chat-fab" id="chatFab" onclick="toggleChat()" aria-label="채팅 열기">💬</button>
  <div class="chat-widget hidden" id="chatWidget">
    <div class="chat-widget-header">
      <h3>MoveOps 배송 도우미</h3>
      <button onclick="toggleChat()" aria-label="채팅 닫기">✕</button>
    </div>
    <div class="chat-mode-toggle" id="chatModeToggle">
      <button type="button" class="chat-mode-btn active" data-mode="ai" onclick="setChatMode('ai')">AI 채팅</button>
      <button type="button" class="chat-mode-btn" data-mode="local" onclick="setChatMode('local')">내 로컬 채팅</button>
    </div>
    <div class="chat-log" id="chatLog">
      <div class="chat-bubble bot">안녕하세요! 배송 요청을 대화로 도와드릴게요. 출발지, 도착지, 보내실 물건을 알려주세요 🙂</div>
    </div>
    <form class="chat-widget-form" id="chatForm">
      <input id="chatInput" placeholder="메시지를 입력하세요" autocomplete="off">
      <button type="submit">전송</button>
    </form>
  </div>

  <script>
    const byId = (id) => document.getElementById(id);
    let deliveryMap = null;
    let pickupMarker = null;
    let dropoffMarker = null;
    let pickupOverlay = null;
    let dropoffOverlay = null;
    let routeLine = null;
    let geocoder = null;
    let mapMode = "pickup";

    function locationValue(kind) {
      return {
        latitude: Number(byId(`${kind}Lat`).value),
        longitude: Number(byId(`${kind}Lng`).value)
      };
    }

    function setMapMode(mode) {
      mapMode = mode;
      document.querySelectorAll(".map-mode[data-mode]").forEach(button => {
        button.classList.toggle("active", button.dataset.mode === mode);
      });
    }

    function mapLabel(text, kind) {
      const element = document.createElement("div");
      element.className = `map-label ${kind}`;
      element.textContent = text;
      return element;
    }

    function distanceKilometers(start, end) {
      const toRadians = value => value * Math.PI / 180;
      const latDelta = toRadians(end.latitude - start.latitude);
      const lngDelta = toRadians(end.longitude - start.longitude);
      const startLat = toRadians(start.latitude);
      const endLat = toRadians(end.latitude);
      const a = Math.sin(latDelta / 2) ** 2
        + Math.cos(startLat) * Math.cos(endLat) * Math.sin(lngDelta / 2) ** 2;
      return 6371 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    }

    function updateDistance() {
      const pickup = locationValue("pickup");
      const dropoff = locationValue("dropoff");
      if (![pickup.latitude, pickup.longitude, dropoff.latitude, dropoff.longitude].every(Number.isFinite)) {
        byId("routeDistance").textContent = "좌표를 입력하면 직선거리를 계산합니다.";
        return;
      }
      const distance = distanceKilometers(pickup, dropoff);
      byId("routeDistance").textContent = `· 직선거리 약 ${distance.toFixed(distance < 10 ? 1 : 0)}km`;
    }

    function renderRoute(shouldFit = false) {
      updateDistance();
      if (!deliveryMap || !window.kakao?.maps) return;

      const pickup = locationValue("pickup");
      const dropoff = locationValue("dropoff");
      if (![pickup.latitude, pickup.longitude, dropoff.latitude, dropoff.longitude].every(Number.isFinite)) return;

      const pickupPosition = new kakao.maps.LatLng(pickup.latitude, pickup.longitude);
      const dropoffPosition = new kakao.maps.LatLng(dropoff.latitude, dropoff.longitude);
      pickupMarker.setPosition(pickupPosition);
      dropoffMarker.setPosition(dropoffPosition);
      pickupOverlay.setPosition(pickupPosition);
      dropoffOverlay.setPosition(dropoffPosition);
      routeLine.setPath([pickupPosition, dropoffPosition]);

      if (shouldFit) {
        const bounds = new kakao.maps.LatLngBounds();
        bounds.extend(pickupPosition);
        bounds.extend(dropoffPosition);
        deliveryMap.setBounds(bounds, 70, 70, 70, 70);
      }
    }

    function fitRoute() {
      renderRoute(true);
    }

    function updateLocation(kind, latitude, longitude, address = "") {
      byId(`${kind}Lat`).value = Number(latitude).toFixed(7);
      byId(`${kind}Lng`).value = Number(longitude).toFixed(7);
      if (address) byId(`${kind}Address`).value = address;
      renderRoute(false);
    }

    function reverseGeocode(kind, latitude, longitude) {
      if (!geocoder) return;
      geocoder.coord2Address(longitude, latitude, (results, status) => {
        if (status !== kakao.maps.services.Status.OK || !results.length) return;
        const result = results[0];
        const address = result.road_address?.address_name || result.address?.address_name || "";
        if (address) byId(`${kind}Address`).value = address;
      });
    }

    function searchAddress(kind) {
      if (!geocoder) {
        showResult("지도 검색 실패", "카카오 지도가 아직 준비되지 않았습니다.", true);
        return;
      }
      const address = byId(`${kind}Address`).value.trim();
      if (!address) return;
      geocoder.addressSearch(address, (results, status) => {
        if (status !== kakao.maps.services.Status.OK || !results.length) {
          showResult("주소 검색 실패", `"${address}" 주소를 찾지 못했습니다.`, true);
          return;
        }
        updateLocation(kind, Number(results[0].y), Number(results[0].x), results[0].address_name);
        setMapMode(kind);
        renderRoute(true);
      });
    }

    function loadKakaoMapSdk(key) {
      return new Promise((resolve, reject) => {
        if (window.kakao?.maps) {
          kakao.maps.load(resolve);
          return;
        }
        const script = document.createElement("script");
        script.src = `https://dapi.kakao.com/v2/maps/sdk.js?appkey=${encodeURIComponent(key)}&libraries=services&autoload=false`;
        script.async = true;
        script.onload = () => kakao.maps.load(resolve);
        script.onerror = () => reject(new Error("카카오 지도 SDK를 불러오지 못했습니다."));
        document.head.appendChild(script);
      });
    }

    async function initializeMap(key) {
      const message = byId("mapMessage");
      if (!key) {
        message.innerHTML = "지도 JavaScript 키가 없습니다.<br><code>KAKAO_JAVASCRIPT_KEY</code>를 설정해주세요.";
        updateDistance();
        return;
      }
      try {
        await loadKakaoMapSdk(key);
        const pickup = locationValue("pickup");
        const dropoff = locationValue("dropoff");
        const pickupPosition = new kakao.maps.LatLng(pickup.latitude, pickup.longitude);
        const dropoffPosition = new kakao.maps.LatLng(dropoff.latitude, dropoff.longitude);
        deliveryMap = new kakao.maps.Map(byId("map"), {
          center: pickupPosition,
          level: 6
        });
        pickupMarker = new kakao.maps.Marker({map: deliveryMap, position: pickupPosition});
        dropoffMarker = new kakao.maps.Marker({map: deliveryMap, position: dropoffPosition});
        pickupOverlay = new kakao.maps.CustomOverlay({
          map: deliveryMap,
          content: mapLabel("출발", "pickup"),
          position: pickupPosition,
          yAnchor: 2.15
        });
        dropoffOverlay = new kakao.maps.CustomOverlay({
          map: deliveryMap,
          content: mapLabel("도착", "dropoff"),
          position: dropoffPosition,
          yAnchor: 2.15
        });
        routeLine = new kakao.maps.Polyline({
          map: deliveryMap,
          strokeWeight: 5,
          strokeColor: "#315efb",
          strokeOpacity: 0.82,
          strokeStyle: "shortdash"
        });
        geocoder = new kakao.maps.services.Geocoder();
        kakao.maps.event.addListener(deliveryMap, "click", event => {
          const latitude = event.latLng.getLat();
          const longitude = event.latLng.getLng();
          updateLocation(mapMode, latitude, longitude);
          reverseGeocode(mapMode, latitude, longitude);
        });
        message.classList.add("hidden");
        renderRoute(true);
      } catch (error) {
        message.innerHTML = `${error.message}<br>카카오 디벨로퍼스에 <b>${location.origin}</b> 도메인이 등록됐는지 확인해주세요.`;
        updateDistance();
      }
    }

    function draft() {
      return {
        orderType: byId("orderType").value,
        productSize: byId("productSize").value,
        pickup: {
          location: {
            basicAddress: byId("pickupAddress").value.trim(),
            latitude: Number(byId("pickupLat").value),
            longitude: Number(byId("pickupLng").value)
          },
          contact: {
            name: byId("pickupName").value.trim(),
            phone: byId("pickupPhone").value.trim()
          }
        },
        dropoff: {
          location: {
            basicAddress: byId("dropoffAddress").value.trim(),
            latitude: Number(byId("dropoffLat").value),
            longitude: Number(byId("dropoffLng").value)
          },
          contact: {
            name: byId("dropoffName").value.trim(),
            phone: byId("dropoffPhone").value.trim()
          }
        },
        productName: byId("productName").value.trim(),
        declaredValue: Number(byId("declaredValue").value),
        paymentType: "CARD",
        waypoints: []
      };
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        ...options,
        headers: {"Content-Type": "application/json", ...(options.headers || {})}
      });
      let body;
      try { body = await response.json(); }
      catch { body = {message: await response.text()}; }
      if (!response.ok) {
        const detail = body.message || body.detail || `HTTP ${response.status}`;
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      return body;
    }

    function showResult(title, value, error = false) {
      const meta = byId("resultMeta");
      const grid = byId("resultGrid");
      const body = byId("resultBody");
      byId("resultTitle").textContent = title;
      meta.innerHTML = "";
      grid.innerHTML = "";

      if (value && typeof value === "object" && value.mode === "cards") {
        const requestId = value.requestId || "-";
        const extraChip = value.highlightText
          ? `<span class="result-chip">${value.highlightText}</span>`
          : `<span class="result-chip">${value.items.length}개 옵션</span>`;
        meta.innerHTML = `<span>요청 ID: <b>${requestId}</b></span>${extraChip}`;
        grid.innerHTML = value.items.map(item => `
          <article class="fare-card${item.selected ? " selected" : ""}">
            ${item.selected ? '<span class="fare-tag">가격 조회 선택값</span>' : ""}
            <h4 class="fare-title">${item.orderTypeLabel}<span>${item.fleetLabel}</span></h4>
            <p class="fare-sub">${item.orderType} · ${item.fleet}</p>
            <div class="fare-row"><span class="fare-label">예상 시간</span><span class="fare-value">${item.timeText}</span></div>
            <div class="fare-row"><span class="fare-label">총 요금</span><span class="fare-value">${item.fareText}</span></div>
            <div class="fare-row"><span class="fare-label">할인</span><span class="fare-value">${item.discountText}</span></div>
          </article>
        `).join("");
        body.textContent = "";
        body.className = "result-body hidden";
      } else {
        body.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
        body.className = error ? "result-body error" : "result-body";
      }
      byId("result").classList.add("visible");
    }

    function secondsToKorean(seconds) {
      const total = Math.max(0, Number(seconds) || 0);
      const hours = Math.floor(total / 3600);
      const minutes = Math.floor((total % 3600) / 60);
      const secs = Math.floor(total % 60);
      if (hours > 0) return `${hours}시간 ${minutes}분`;
      if (minutes > 0) return `${minutes}분 ${secs}초`;
      return `${secs}초`;
    }

    function won(value) {
      return `${Number(value || 0).toLocaleString("ko-KR")}원`;
    }

    function orderTypeLabel(orderType) {
      return {
        QUICK: "퀵",
        QUICK_ECONOMY: "퀵 이코노미",
        QUICK_EXPRESS: "퀵 급송",
        DOBO: "도보 배송"
      }[orderType] || orderType;
    }

    function fleetLabel(fleet) {
      return {
        MOTORCYCLE: "오토바이",
        PASSENGER_CAR: "승용차",
        DAMAS: "다마스",
        LABO: "라보",
        TON: "1톤"
      }[fleet] || fleet;
    }

    function formatEstimateResult(data) {
      const rows = Array.isArray(data?.lists) ? data.lists : [];
      if (!rows.length) return JSON.stringify(data, null, 2);
      return {
        mode: "cards",
        requestId: data.requestId || "-",
        items: rows
          .slice()
          .sort((a, b) => (Number(a.totalFareAmount) || 0) - (Number(b.totalFareAmount) || 0))
          .map(item => ({
            orderType: item.orderType,
            fleet: item.fleet,
            orderTypeLabel: orderTypeLabel(item.orderType),
            fleetLabel: fleetLabel(item.fleet),
            timeText: secondsToKorean(item.estimatedTime),
            fareText: won(item.totalFareAmount),
            discountText: won(item.discountAmount),
            selected: false
          }))
      };
    }

    function formatPriceResult(priceData, estimateData, selectedOrderType) {
      const rows = Array.isArray(estimateData?.lists) ? estimateData.lists : [];
      const selectedPrice = Number(priceData?.totalPrice ?? NaN);
      if (!rows.length) {
        if (!Number.isFinite(selectedPrice)) return JSON.stringify(priceData, null, 2);
        return {
          mode: "cards",
          requestId: priceData?.requestId || "-",
          highlightText: `선택 요금 ${won(selectedPrice)}`,
          items: [{
            orderType: selectedOrderType,
            fleet: "-",
            orderTypeLabel: orderTypeLabel(selectedOrderType),
            fleetLabel: "요금 조회",
            timeText: "-",
            fareText: won(selectedPrice),
            discountText: won(0),
            selected: true
          }]
        };
      }

      const pickIndex = rows.findIndex(item => {
        if (item.orderType !== selectedOrderType) return false;
        if (!Number.isFinite(selectedPrice)) return true;
        return Number(item.totalFareAmount) === selectedPrice;
      });

      return {
        mode: "cards",
        requestId: priceData?.requestId || estimateData?.requestId || "-",
        highlightText: Number.isFinite(selectedPrice)
          ? `선택 요금 ${won(selectedPrice)}`
          : `${rows.length}개 옵션`,
        items: rows
          .slice()
          .sort((a, b) => (Number(a.totalFareAmount) || 0) - (Number(b.totalFareAmount) || 0))
          .map((item, index) => ({
            orderType: item.orderType,
            fleet: item.fleet,
            orderTypeLabel: orderTypeLabel(item.orderType),
            fleetLabel: fleetLabel(item.fleet),
            timeText: item.estimatedTime ? secondsToKorean(item.estimatedTime) : "-",
            fareText: won(item.totalFareAmount),
            discountText: won(item.discountAmount),
            selected: index === pickIndex
          }))
      };
    }

    async function withButton(button, job) {
      button.disabled = true;
      try { await job(); }
      catch (error) { showResult("요청 실패", error.message, true); }
      finally { button.disabled = false; }
    }

    async function callEstimate(button) {
      await withButton(button, async () => {
        const result = await api("/api/deliveries/estimate", {method: "POST", body: JSON.stringify(draft())});
        showResult("예상시간 조회 결과", formatEstimateResult(result.data));
      });
    }

    async function callPrice(button) {
      await withButton(button, async () => {
        const requestDraft = draft();
        const priceResult = await api("/api/deliveries/price", {method: "POST", body: JSON.stringify(requestDraft)});
        let estimateResult = null;
        try {
          estimateResult = await api("/api/deliveries/estimate", {method: "POST", body: JSON.stringify(requestDraft)});
        } catch {
          // 가격 API는 성공했지만 옵션 조회가 실패할 수 있으므로 단일 카드로라도 보여준다.
        }
        showResult(
          "가격 조회 결과",
          formatPriceResult(priceResult.data, estimateResult?.data, requestDraft.orderType)
        );
      });
    }

    async function createOrder(button) {
      if (!confirm("Sandbox 테스트 주문만 허용됩니다. 계속할까요?")) return;

      const pickupPhone = byId("pickupPhone").value.trim();
      const dropoffPhone = byId("dropoffPhone").value.trim();
      const sandboxPhone = /^010-1000-\d{4}$/;
      if (!sandboxPhone.test(pickupPhone) || !sandboxPhone.test(dropoffPhone)) {
        showResult(
          "주문 차단",
          "테스트 전화번호 형식(010-1000-XXXX)만 허용됩니다. 실제 번호 입력 시 접수되지 않습니다.",
          true
        );
        return;
      }

      const phrase = prompt("실제 주문이 아님을 확인했습니다. 계속하려면 SANDBOX 를 입력하세요.", "");
      if ((phrase || "").trim().toUpperCase() !== "SANDBOX") {
        showResult("주문 취소", "확인 문구가 일치하지 않아 주문을 전송하지 않았습니다.", true);
        return;
      }

      await withButton(button, async () => {
        const result = await api("/api/orders", {method: "POST", body: JSON.stringify(draft())});
        const id = result.data.partnerOrderId || result.data.order?.partnerOrderId;
        if (id) byId("trackingId").value = id;
        showResult("주문 접수 결과", result.data);
        await loadOrders();
      });
    }

    async function trackOrder(refresh = false) {
      const id = byId("trackingId").value.trim();
      if (!id) return;
      try {
        const result = await api(`/api/orders/${encodeURIComponent(id)}?refresh=${refresh}`);
        byId("trackBody").textContent = JSON.stringify(result.data, null, 2);
        byId("trackBody").className = "";
        byId("trackResult").classList.add("visible");
        await loadOrders();
      } catch (error) {
        byId("trackBody").textContent = error.message;
        byId("trackBody").className = "error";
        byId("trackResult").classList.add("visible");
      }
    }

    async function refreshProvider() { await trackOrder(true); }

    async function cancelTrackedOrder() {
      const id = byId("trackingId").value.trim();
      if (!id || !confirm("이 Sandbox 주문을 취소할까요?")) return;
      try {
        await api(`/api/orders/${encodeURIComponent(id)}/cancel`, {method: "PATCH"});
        await trackOrder();
      } catch (error) {
        byId("trackBody").textContent = error.message;
        byId("trackBody").className = "error";
        byId("trackResult").classList.add("visible");
      }
    }

    async function loadOrders() {
      const list = byId("orderList");
      try {
        const result = await api("/api/orders?limit=20");
        if (!result.data.length) {
          list.innerHTML = '<div class="empty">아직 저장된 주문이 없습니다.</div>';
          return;
        }
        list.innerHTML = result.data.map(order => `
          <div class="order" onclick="selectOrder('${order.partnerOrderId.replaceAll("'", "\\'")}')">
            <div class="order-top">
              <span class="order-id">${order.partnerOrderId}</span>
              <span class="badge">${order.status}</span>
            </div>
            <time>${new Date(order.updatedAt).toLocaleString("ko-KR")}</time>
          </div>`).join("");
      } catch (error) {
        list.innerHTML = `<div class="empty error">${error.message}</div>`;
      }
    }

    function selectOrder(id) {
      byId("trackingId").value = id;
      trackOrder();
    }

    const chatSessionId = (() => {
      let id = localStorage.getItem("moveops_session_id");
      if (!id) {
        id = crypto.randomUUID();
        localStorage.setItem("moveops_session_id", id);
      }
      return id;
    })();

    function toggleChat() {
      byId("chatWidget").classList.toggle("hidden");
    }

    let chatMode = "ai";

    function setChatMode(mode) {
      chatMode = mode;
      document.querySelectorAll(".chat-mode-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.mode === mode);
      });
    }

    function appendChatBubble(role, text) {
      const bubble = document.createElement("div");
      bubble.className = `chat-bubble ${role}`;
      bubble.textContent = text;
      const log = byId("chatLog");
      log.appendChild(bubble);
      log.scrollTop = log.scrollHeight;
    }

    async function sendChatMessage() {
      const input = byId("chatInput");
      const message = input.value.trim();
      if (!message) return;
      appendChatBubble("user", message);
      input.value = "";
      try {
        const result = await api("/api/agent/chat", {
          method: "POST",
          headers: {"X-Session-Id": chatSessionId},
          body: JSON.stringify({message, mode: chatMode})
        });
        appendChatBubble("bot", result.data?.reply || "응답을 받았지만 표시할 내용이 없어요.");
      } catch (error) {
        appendChatBubble("bot", `연결에 문제가 있었어요: ${error.message}. 잠시 후 다시 시도하거나 왼쪽 폼으로 접수해보세요.`);
      }
    }

    byId("chatForm").addEventListener("submit", (event) => {
      event.preventDefault();
      sendChatMessage();
    });

    async function initialize() {
      try {
        const config = await api("/api/config");
        byId("status").innerHTML = config.data.configured
          ? '카카오 연동 <b>설정됨</b>'
          : '카카오 연동 <span class="error">미설정</span>';
        await initializeMap(config.data.kakaoJavascriptKey);
      } catch {
        byId("status").textContent = "설정 확인 실패";
        byId("mapMessage").textContent = "지도 설정을 확인하지 못했습니다.";
      }
      await loadOrders();
    }
    ["pickupLat", "pickupLng", "dropoffLat", "dropoffLng"].forEach(id => {
      byId(id).addEventListener("change", () => renderRoute(false));
    });
    updateDistance();
    initialize();
  </script>
</body>
</html>
"""
