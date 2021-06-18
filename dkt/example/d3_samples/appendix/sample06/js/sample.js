var svgWidth = 320;	// SVG 요소의 넓이
var svgHeight = 320;	// SVG 요소의 높이
// 데이터셋
var dataSet = {
	name : "전국", value : 128057352,
	children : [
		{ name : "도쿄도", value : 13159388 },
		{ name : "오사카부", value : 8865245 },
		{ name : "아이치현", value : 7410719 },
		{ name : "나가노현", value : 2152449,
			children : [
				{ name : "나가노시", value : 381511 },
				{ name : "마츠모토시", value : 243037 },
				{ name : "시오지리시", value : 67670 }
			]
		}
	]
}
// 색 준비
var color = d3.scale.category10();  // 10색을 지정
// 팩 레이아웃
var bubble = d3.layout.pack()
  .size([320, 320])	// 표시 크기를 지정
// 팩 레이아웃에 사용할 그룹을 작성
var pack = d3.select("#myGraph")
  .selectAll("g")
  .data(bubble.nodes(dataSet))	// 데이터셋을 요소에 설정
  .enter()
  .append("g")
  .attr("transform", function(d, i){
		return "translate(" + d.x + "," + d.y + ")";	// X,Y 좌표를 설정
	})
// 원을 생성
var circle = pack.append("circle")	// circle 요소 추가
  .attr("r", function(d){	// 반지름을 지정
		return d.r;
	})
  .style("fill", function(d, i){
		return color(i);
	})
// 원에 표시할 문자를 생성
pack.append("text")
  .style("opacity", 1.0)	// 불투명으로 함
  .text(function(d, i){
		if (d.depth == 1){	// 제1계층(도도부현 수준)만 대상
			return d.name;	// name 속성 내용을 반환
		}
		return null;	// 제1계층 이외는 표시하지 않음
	})
// 피시아이 설정
var fisheye = d3.fisheye.circular()
  .radius(100)
  .distortion(3)
// 마우스 이동으로 효과를 시작하도록 함
d3.select("#myGraph").on("mousemove", function(){
	fisheye.focus(d3.mouse(this));	// 마우스 좌표를 확대 중심에 설정
	circle.each(function(d, i){	// 확대 대산 요소에 대해 처리
		d.fisheye = fisheye(d);	// 계산한 결과를 fisheye 객체의 x, y, z 속성에 입력
		d.r = d.r;	// 계산상 원래의 반지름을 r 속성에 대입
	})
  .attr("r", function(d) { return d.fisheye.z * d.r; })	// 확대율과 반지름을 곱한 새로운 반지름으로 함
})
