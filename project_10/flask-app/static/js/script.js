document.getElementById("button1").addEventListener("click", function() {
     document.getElementById("wrn").innerHTML = "<i>Validating, please wait!</i>";
    var form = document.getElementById("inputform");
    var elements = form.elements;
    for (var i = 0, len = elements.length; i < len; ++i) {
        elements[i].readOnly = true;
}
}); 


document.getElementById("button2").addEventListener("click", function() {
     document.getElementById("wrn").innerHTML = "<i>Please wait!</i>";
    var form = document.getElementById("inputform");
    var elements = form.elements;
    for (var i = 0, len = elements.length; i < len; ++i) {
        elements[i].readOnly = true;
}
});