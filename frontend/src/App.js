import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [stationName, setStationName] = useState('');
  const [dayOfWeek, setDayOfWeek] = useState('');
  const [timeSlot, setTimeSlot] = useState('');
  const [result, setResult] = useState(null);

  const handlePredict = async () => {
    try {
      const response = await axios.post('http://localhost:8000/api/predict/', {
        stationName,
        dayOfWeek,
        timeSlot
      });
      setResult(response.data.result);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="App">
      <h1>혼잡도 예측</h1>
      <div className="form-group">
        <label>역명</label>
        <input
          type="text"
          value={stationName}
          onChange={(e) => setStationName(e.target.value)}
        />
      </div>
      <div className="form-group">
        <label>요일</label>
        <input
          type="text"
          value={dayOfWeek}
          onChange={(e) => setDayOfWeek(e.target.value)}
        />
      </div>
      <div className="form-group">
        <label>시간대</label>
        <input
          type="text"
          value={timeSlot}
          onChange={(e) => setTimeSlot(e.target.value)}
        />
      </div>
      <button onClick={handlePredict}>예측</button>
      {result && <h2>예측 결과: {result}</h2>}
    </div>
  );
}

export default App;