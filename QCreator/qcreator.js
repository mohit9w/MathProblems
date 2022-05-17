// Example starter JavaScript for disabling form submissions if there are invalid fields
var selectedOptions = 0;
$(function () {
  'use strict'
  enableFormValidations();

  $('#addedOptionDiv').on('DOMSubtreeModified', function(){
    if(selectedOptions == 4){
        $('#option').removeAttr('required');
    } else {
        $('#option').prop('required', true);
    }
  });
});
function enableFormValidations(){
    // Fetch all the forms we want to apply custom Bootstrap validation styles to
    var forms = document.querySelectorAll('.needs-validation')

    // Loop over them and prevent submission
    Array.prototype.slice.call(forms)
    .forEach(function (form) {
      form.addEventListener('submit', function (event) {
        if (!form.checkValidity()) {
          event.preventDefault()
          event.stopPropagation()
        }

        form.classList.add('was-validated')
      }, false)
    });
}

function saveOption(){
    if(selectedOptions != 4 && $('#option').val().length > 0){
        let option = $('#option').val();
        selectedOptions++;
        $('#addedOptionDiv').append("<button class=\"badge bg-primary text-wrap fw-light\" style=\"width: 6rem;\" id=\""+ option +"\" onClick=\"deleteOption('"+ option +"')\">" + option + "<br>click to remove</button>");
        $('#option').val('');
    }
}
function deleteOption(deleteme){
    selectedOptions--;
    $('#'+deleteme).remove();
}