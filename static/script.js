document.addEventListener('DOMContentLoaded', () => {
    const languageSelect = document.getElementById('languageSelect');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const clearBtn = document.getElementById('clearBtn');
    const backspaceBtn = document.getElementById('backspaceBtn');
    const speakRecogBtn = document.getElementById('speakRecogBtn');
    const speakTransBtn = document.getElementById('speakTransBtn');
    const recognizedTextOutput = document.getElementById('recognised-text-output');
    const translatedTextOutput = document.getElementById('translated-text-output');

    // Event listeners
    languageSelect.addEventListener('change', (event) => {
        setLanguage(event.target.value);
    });

    startBtn.addEventListener('click', () => {
        startRecognition();
        startBtn.disabled = true; // Disable the start button to prevent multiple starts
        stopBtn.disabled = false;  // Enable the stop button
    });

    stopBtn.addEventListener('click', () => {
        stopRecognition();
        startBtn.disabled = false; // Enable the start button
        stopBtn.disabled = true;   // Disable the stop button
    });

    clearBtn.addEventListener('click', clearText);
    backspaceBtn.addEventListener('click', backspaceText);
    speakRecogBtn.addEventListener('click', () => speakText(recognizedTextOutput.value));
    speakTransBtn.addEventListener('click', () => speakText(translatedTextOutput.value));

    function setLanguage(language) {
        fetch('/set_language', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ language: language }),
        })
        .then(response => response.json())
        .then(data => console.log(data.message));
    }

    function startRecognition() {
        fetch('/start_recognition', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => console.log(data.message));

        // Start polling for recognition results
        recognitionInterval = setInterval(pollRecognition, 3000); // Poll every 3 seconds
    }

    function stopRecognition() {
        fetch('/stop_recognition', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            console.log(data.message);
            clearInterval(recognitionInterval); // Stop polling
        });
    }

    function pollRecognition() {
        fetch('/recognize', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            recognizedTextOutput.value = data.recognized || '';
            translatedTextOutput.value = data.translated || '';
        });
    }

    function clearText() {
        recognizedTextOutput.value = '';
        translatedTextOutput.value = '';
        fetch('/clear', { method: 'POST' }); // Clear on the server side
    }

    function backspaceText() {
        fetch('/backspace', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            recognizedTextOutput.value = data.recognized || '';
        });
    }

    function speakText(text) {
        if (text) {
            const utterance = new SpeechSynthesisUtterance(text);
            window.speechSynthesis.speak(utterance);
        }
    }
});
