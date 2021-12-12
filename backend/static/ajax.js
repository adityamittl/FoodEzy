function addtodo(id){
    $.ajax({
        type: "POST",
        url: "/addexercise",
        headers: { 'X-CSRFToken': Cookies.get('csrftoken') },
        data: {
          'eid': id 
        },
        success: function () {
          alert('Exercise has be successfully added to your Todo')
        }
      });
}