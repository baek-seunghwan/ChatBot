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
    display: block;
    flex: 0 0 auto;
    color: #10254d;
    text-decoration: none;
  }
  .movb-brand-lockup {
    display: block;
    width: 108px;
    height: 36px;
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
    .movb-brand-lockup {
      width: 96px;
      height: 32px;
    }
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
  <a class="movb-brand" href="/" aria-label="MOVB · Move Better Together 홈">
    <svg class="movb-brand-lockup" viewBox="0 0 300 100" aria-hidden="true">
      <text x="0" y="72" textLength="300" lengthAdjust="spacingAndGlyphs"
        fill="#10254d" font-family="Arial Black, Arial, sans-serif"
        font-size="88" font-weight="900">MOVB</text>
      <text x="0" y="98" textLength="300" lengthAdjust="spacingAndGlyphs"
        fill="#7b8089" font-family="Arial, sans-serif"
        font-size="21" font-weight="800">MOVE BETTER, TOGETHER</text>
    </svg>
  </a>
  <nav class="movb-primary-nav" aria-label="주요 메뉴">
    <a class="movb-nav-item{active_class("about")}" data-nav="about"
      href="/about">브랜드 소개</a>
    <a class="movb-nav-item{active_class("bundle")}" data-nav="bundle"
      href="/order">퀵 접수하기</a>
    <a class="movb-nav-item{active_class("history")}" data-nav="history"
      href="/history">이용 내역</a>
  </nav>
  <a class="movb-login-button" id="loginButton" href="/order?login=1"
    aria-haspopup="dialog" aria-controls="authModal" aria-expanded="false"
    onclick="if (typeof openAuthModal === 'function') {{
      event.preventDefault(); openAuthModal();
    }}">로그인</a>
  <span class="movb-visually-hidden" id="serviceStatus" aria-live="polite">
    연동 상태 확인 중
  </span>
</header>
"""
