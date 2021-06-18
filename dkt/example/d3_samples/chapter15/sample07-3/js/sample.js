var canvasWidth = 640;	// Canvas 요소의 넓이
var canvasHeight = 640;	// Canvas 요소의 높이
var degree = 0;	// 회전 각도
var earthSize = 280;	// 지구의 크기
var earth = d3.geo.orthographic()   // 투영 방법을 Orthographic로 설정
  .translate([canvasWidth/2, canvasHeight/2])	//화면에서의 표시 위치 조정
  .clipAngle(90)	// 클립 범위 지정
  .scale(earthSize)	// 축척 지정
  .rotate([degree, -25])	// 회전 각도 지정
var path = d3.geo.path()	// 패스와 투영 방법 설정
  .projection(earth)
var context = d3.select("#myGraph").node().getContext("2d");	// 컨텍스트 얻기
// 지구의 데이터를 불러옴
d3.json("data/world.json", function(error, world) {
	d3.timer(function(){
		earth.rotate([degree, -25]);	// 각도 설정
		degree = degree + 2;	// 2도씩 움직임
		// 이하가 Canvas 그리기 처리
		context.clearRect(0, 0, canvasWidth, canvasHeight);	// Canvas 안 삭제
		context.fillStyle = "#22f";	// 칠하기 색을 지정
		context.beginPath();
		context.arc(canvasWidth/2, canvasHeight/2, earthSize, 0, Math.PI*2, 1);	// 지구(원) 그리기
		context.fill();	// 지구를 칠함
		context.fillStyle = "#eee";	// 칠하기 색을 지정
		context.strokeStyle = "black";	// 선의 색을 지정
		context.lineWidth = 0.5;	// 선의 굵기
		context.beginPath();
		path.context(context)(world);	// 지도의 패스 생성
		context.fill();	// 지도를 칠함
		context.stroke();	// 지도의 경계선 그리기
	});
})
