<script>

async function getSkills() {

    const role = document.getElementById("roleInput").value;

    const response = await fetch("/get_skills", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({role: role})
    });

    const data = await response.json();

    document.getElementById("results").innerHTML = data.skills;

}

</script>