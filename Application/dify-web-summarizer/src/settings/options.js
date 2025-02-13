document.getElementById('save-btn').addEventListener('click', function() {
  const apiKey = document.getElementById('api-key').value;
  if (apiKey) {
    chrome.storage.sync.set({ apiKey: apiKey }, function() {
      document.getElementById('status').textContent = 'API Key saved successfully!';
    });
  } else {
    document.getElementById('status').textContent = 'Please enter a valid API Key.';
  }
});

// 讀取已儲存的 API Key
chrome.storage.sync.get('apiKey', function(data) {
  if (data.apiKey) {
    document.getElementById('api-key').value = data.apiKey;
  }
});