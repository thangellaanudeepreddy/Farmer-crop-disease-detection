function App() {
  return (
    <div style={{ textAlign: "center", padding: "40px" }}>
      <h1>🌱 Farmer Crop Disease Detection</h1>
      <p>Upload a crop leaf image to detect diseases using AI.</p>

      <input type="file" accept="image/*" />

      <br /><br />

      <button>Detect Disease</button>
    </div>
  );
}

export default App;