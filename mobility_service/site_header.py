from __future__ import annotations


SITE_HEADER_CSS = """
<style id="movb-shared-header-styles">
  .movb-site-header {
    position: sticky;
    z-index: 50;
    top: 0;
    display: flex;
    align-items: center;
    gap: 22px;
    min-height: 64px;
    padding: 9px clamp(18px, 4vw, 56px);
    border-bottom: 1px solid rgba(229, 231, 235, .9);
    background: rgba(255, 255, 255, .96);
    color: #17191f;
    backdrop-filter: blur(16px);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo",
      "Noto Sans KR", sans-serif;
    font-size: 16px;
    line-height: 1.4;
  }
  .movb-brand {
    display: flex;
    flex: 0 0 auto;
    align-items: center;
    gap: 8px;
    color: inherit;
    text-decoration: none;
  }
  .movb-brand-mark {
    display: grid;
    width: 38px;
    height: 38px;
    place-items: center;
    border-radius: 10px;
    border: 1px solid #e7e9ef;
    background: #ffffff;
    overflow: hidden;
  }
  .movb-brand-mark svg { width: 100%; height: 100%; display: block; }
  .movb-brand-copy {
    display: grid;
    gap: 0;
    color: #17191f;
    font-weight: 850;
    line-height: 1.02;
    letter-spacing: -.02em;
  }
  .movb-brand-copy span { font-size: 20px; }
  .movb-brand-copy small {
    color: #737986;
    font-size: 8px;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
  }
  .movb-brand-copy .movb-brand-tagline {
    color: #9aa0ab;
    font-size: 8px;
    font-weight: 700;
    letter-spacing: .10em;
  }
  .movb-primary-nav {
    display: flex;
    align-items: center;
    gap: 3px;
  }
  .movb-nav-item,
  .movb-login-button {
    display: inline-flex;
    min-height: 38px;
    align-items: center;
    justify-content: center;
    border: 0;
    border-radius: 6px;
    padding: 8px 12px;
    background: transparent;
    color: #17191f;
    box-shadow: none;
    font: inherit;
    font-size: 13px;
    font-weight: 800;
    text-decoration: none;
    white-space: nowrap;
  }
  .movb-nav-item:hover,
  .movb-login-button:hover {
    background: #f3f4f7;
    color: #17191f;
  }
  .movb-nav-item.active {
    background: #10254d;
    color: #fff;
  }
  .movb-login-button {
    margin-left: auto;
    border: 1px solid #e2e5eb;
  }
  .movb-login-button.logged-in {
    border-color: rgba(255, 204, 0, .8);
    background: rgba(255, 204, 0, .22);
    color: #6a5400;
  }
  .movb-visually-hidden {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
  @media (max-width: 860px) {
    .movb-site-header {
      flex-wrap: wrap;
      gap: 8px;
      padding: 8px 14px;
    }
    .movb-login-button { margin-left: auto; }
    .movb-primary-nav {
      order: 3;
      width: 100%;
      overflow-x: auto;
      padding-bottom: 1px;
      scrollbar-width: none;
    }
    .movb-primary-nav::-webkit-scrollbar { display: none; }
    .movb-nav-item { flex: 0 0 auto; }
  }
  @media (max-width: 560px) {
    .movb-brand-copy small { display: none; }
    .movb-nav-item {
      min-height: 36px;
      padding: 7px 10px;
      font-size: 12px;
    }
  }
</style>
"""


def site_header(active: str = "bundle") -> str:
    def active_class(name: str) -> str:
        return " active" if name == active else ""

    return f"""
<header class="movb-site-header">
  <a class="movb-brand" href="/" aria-label="MOVB 홈">
    <span class="movb-brand-mark" aria-label="MOVB 로고">
      <svg viewBox="0 0 100 100" aria-hidden="true">
        <rect x="0" y="0" width="100" height="100" rx="22" fill="#ffffff"/>
        <path d="M31 69V31L50 51 69 31V69" fill="none" stroke="#10254d"
          stroke-width="10" stroke-linecap="round" stroke-linejoin="round"/>
        <rect x="37" y="76" width="26" height="4" rx="2" fill="#ffcc00"/>
      </svg>
    </span>
    <span class="movb-brand-copy">
      <span>MOVB</span>
      <small>Move Better,</small>
      <small class="movb-brand-tagline">TOGETHER</small>
    </span>
  </a>
  <nav class="movb-primary-nav" aria-label="주요 메뉴">
    <a class="movb-nav-item{active_class("about")}" data-nav="about"
      href="/about">브랜드 소개</a>
    <a class="movb-nav-item{active_class("bundle")}" data-nav="bundle"
      href="/order#smartDelivery"
      onclick="if (typeof openSmartDelivery === 'function') {{
        event.preventDefault(); openSmartDelivery();
      }}">스마트 딜리버리</a>
    <a class="movb-nav-item{active_class("history")}" data-nav="history"
      href="/history">이용 내역</a>
  </nav>
  <a class="movb-login-button" id="loginButton" href="/order?login=1"
    onclick="if (typeof openAuthModal === 'function') {{
      event.preventDefault(); openAuthModal();
    }}">로그인</a>
  <span class="movb-visually-hidden" id="serviceStatus" aria-live="polite">
    연동 상태 확인 중
  </span>
</header>
"""
