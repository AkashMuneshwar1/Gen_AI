function generateText() {
    let prompt = document.getElementById("prompt").value;
    let responseElement = document.getElementById("response");
    let loadingElement = document.getElementById("loading");

    if (!prompt.trim()) {
        alert("Please enter a prompt.");
        return;
    }

    responseElement.innerText = "";
    loadingElement.classList.remove("hidden");

    fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: prompt }),
    })
    .then(response => response.json())
    .then(data => {
        responseElement.innerText = data.generated_text;
        loadingElement.classList.add("hidden");
    })
    .catch(error => {
        responseElement.innerText = "Error: Unable to generate text.";
        loadingElement.classList.add("hidden");
    });
}
