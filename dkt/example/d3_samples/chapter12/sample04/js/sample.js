var svgWidth = 320;	// SVG요소의 넓이
var svgHeight = 320;	// SVG요소의 높이
// JSON 데이터를 불러와 처리
d3.json("data2010.json", function(error, data){
	drawPackLayout(data);
});
// 팩 레이아웃 표시
function drawPackLayout(dataSet){
	// 색을 준비
	var color = d3.scale.category10();  // 10색을 지정
	// 팩 레이아웃
	var bubble = d3.layout.pack()
	  .size([320, 320])	// 표시 크기 지정
	// 팩 레이아웃으로 사용할 크룹을 작성
	var pack = d3.select("#myGraph")
	  .selectAll("g")
	  .data(bubble.nodes(dataSet))	// 데이터셋을 요소에 설정
	  .enter()
	  .append("g")
	  .attr("transform", function(d, i){
			return "translate(" + d.x + "," + d.y + ")";	// X,Y 좌표 설정
		})
	// 원 생성
	pack.append("circle")	// circle요소를 추가
	  .attr("r", 0)	// 최초 반지름은 0으로 함
	  .transition()
	  .duration(function(d, i){	// 자식의 깊이에 따라 대기 시간을 설정
			return d.depth * 1000 + 500;	// 깊이×1초+0.5초
		})
	  .attr("r", function(d){	// 반지름 지정
			return d.r;
		})
	  .style("fill", function(d, i){
			return color(i);
		})
	// 원에 표시할 문자 생성
	pack.append("text")
	  .style("opacity", 0)	// 투명으로 함
	  .transition()
	  .duration(3000)	// 3초에 걸쳐 표시
	  .style("opacity", 1.0)	// 불투명으로 함
	  .text(function(d, i){
			if (d.depth == 1){	// 제1계층(도도부현)의 대상
				return d.name;	// name 속성의 내용을 반환
			}
			return null;	// 제1계층 이외는 표시하지 않음
		})
}
