function callDifyAPI(apiKey, inputUrl) {
  const apiUrl = "https://api.dify.ai/v1/workflows/run";

  const payload = {
    inputs: { input_url: inputUrl },
    response_mode: "blocking",
    user: "dify-web-summarizer"
  };

  return fetch(apiUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`
    },
    body: JSON.stringify(payload)
  })
    .then(response => response.text())
    .then(rawResponse => {
      try {
        const data = JSON.parse(rawResponse);
        return data;
      } catch (error) {
        throw new Error(`The API response is not a valid JSON: ${rawResponse}`);
      }
    });
}