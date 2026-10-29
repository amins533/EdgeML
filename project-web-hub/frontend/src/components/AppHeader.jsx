export function AppHeader() {
  return (
    <header className="app-header">
      <h1 className="app-header__title">EdgeML American Football Analyse</h1>
      <div className="app-header__status">
        Jetson Orin Nano: <span className="app-header__status--online">ONLINE</span>
      </div>
    </header>
  );
}