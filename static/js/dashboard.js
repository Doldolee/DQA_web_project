const Category_link = document.querySelector(".category_link");
const Category_container = document.querySelector(".category-container");
const Category_form = document.querySelector(".category_form");
const Naver_movie = document.querySelectorAll(".naver_movie");
const Daum_movie = document.querySelectorAll(".daum_movie");
const Naver_news = document.querySelectorAll(".naver_news");
const Daum_news = document.querySelectorAll(".daum_news");

const SHOWING_CN = "showing";
let nm_flag = 0;
let nn_flag = 0;
let dm_flag = 0;
let dn_flag = 0;
function clickHandler(event) {
  event.classList.add(SHOWING_CN);
}

function init() {
  Category_link.addEventListener("click", (event) => {
    event.preventDefault();
    Category_container.classList.add(SHOWING_CN);
  });
  Naver_movie[0].addEventListener("click", (event) => {
    event.preventDefault();
    if (nm_flag === 0) {
      Naver_movie[1].classList.add(SHOWING_CN);
      nm_flag = 1;
    } else {
      Naver_movie[1].classList.remove(SHOWING_CN);
      nm_flag = 0;
    }
  });
  Daum_movie[0].addEventListener("click", (event) => {
    event.preventDefault();
    if (dm_flag === 0) {
      Daum_movie[1].classList.add(SHOWING_CN);
      dm_flag = 1;
    } else {
      Daum_movie[1].classList.remove(SHOWING_CN);
      dm_flag = 0;
    }
  });
  Naver_news[0].addEventListener("click", (event) => {
    event.preventDefault();
    if (nn_flag === 0) {
      Naver_news[1].classList.add(SHOWING_CN);
      nn_flag = 1;
    } else {
      Naver_news[1].classList.remove(SHOWING_CN);
      nn_flag = 0;
    }
  });
  Daum_news[0].addEventListener("click", (event) => {
    event.preventDefault();
    if (dn_flag === 0) {
      Daum_news[1].classList.add(SHOWING_CN);
      dn_flag = 1;
    } else {
      Daum_news[1].classList.remove(SHOWING_CN);
      dn_flag = 0;
    }
  });
}
init();
