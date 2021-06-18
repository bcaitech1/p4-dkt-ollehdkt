var svgWidth = 640;	// SVG요소의 넓이
var svgHeight = 640;	// SVG요소의 높이
var path = d3.geo.path()	// 지도의 패스 생성
  .projection(
		d3.geo.mercator()   // 투영 방법을 메르카토르 도법으로 설정
	  .translate([svgWidth/2, svgHeight/2])	// 화면에서의 표시 위치 조정
	  .scale(100)	// 축척 지정
	)
// 세계 지도 데이터를 불러옴
d3.json("data/world.json", function(error, world) {
	d3.select("#myGraph")
	  .selectAll("path")	// path 요소 지정
	  .data(world.features)	// 데이터를 저장
	  .enter()
	  .append("path")	// path를 추가
	  .attr("d", path)	// 지형 데이터 설정
	  .style("fill", function(d, i){
			return "hsl("+i+", 80%, 60%)";	// 색 설정
		})
})
