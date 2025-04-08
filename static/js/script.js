document.addEventListener('DOMContentLoaded', function() {
    // Manejo de donaciones
    const donationForm = document.getElementById('donation-form');
    const amountButtons = document.querySelectorAll('.amount-btn');
    const customAmountInput = document.getElementById('custom-amount');
    const loadingDiv = document.getElementById('loading');
    
    // Selección de monto
    amountButtons.forEach(button => {
        button.addEventListener('click', function() {
            amountButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            customAmountInput.value = '';
        });
    });
    
    customAmountInput.addEventListener('input', function() {
        amountButtons.forEach(btn => btn.classList.remove('active'));
    });
    
    // Envío del formulario
    donationForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        let amount;
        const activeButton = document.querySelector('.amount-btn.active');
        
        if (activeButton) {
            amount = activeButton.dataset.amount;
        } else if (customAmountInput.value && parseFloat(customAmountInput.value) > 0) {
            amount = parseFloat(customAmountInput.value).toFixed(2);
        } else {
            alert('Por favor selecciona o ingresa un monto válido.');
            return;
        }
        
        loadingDiv.style.display = 'flex';
        
        fetch('/create_payment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `amount=${amount}`
        })
        .then(response => response.json())
        .then(data => {
            if (data.redirect_url) {
                window.location.href = data.redirect_url;
            } else if (data.error) {
                alert('Error: ' + data.error);
                loadingDiv.style.display = 'none';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            loadingDiv.style.display = 'none';
        });
    });
    
    // Cambio de idioma
    const langLinks = document.querySelectorAll('.language-switcher a');
    langLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            window.location.href = this.href;
        });
    });
});