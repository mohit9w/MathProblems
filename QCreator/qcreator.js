// Example starter JavaScript for disabling form submissions if there are invalid fields
var selectedOptions = 0;
$(function () {
  'use strict'
  enableFormValidations();
  $('#txtoptiondiv').on('DOMSubtreeModified',function(){if(selectedOptions==4){getTxtOptionField().removeAttr('required');}else{getTxtOptionField().prop('required',true);}});
});

function getTxtOptionField(){return $('#txtoption');}
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

function saveTxtOption(){
    if(selectedOptions != 4 && getTxtOptionField().val().length > 0){
        let option = getTxtOptionField().val();
        selectedOptions++;
        $('#txtoptiondiv').append("<button class=\"badge bg-primary text-wrap fw-light\" style=\"width: 6rem;\" id=\""+ option +"\" onClick=\"deleteOption('"+ option +"')\">" + option + "<br>click to remove</button>");
        getTxtOptionField().val('');
    }
}
function deleteOption(deleteme){
    selectedOptions--;
    $('#'+deleteme).remove();
}