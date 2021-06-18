var svgWidth = 640;	// SVG요소의 넓이
var svgHeight = 640;	// SVG요소의 높이
var degree = 0;	// 회전 각도
var earth = d3.geo.orthographic()   // 투영 방법을 Orthographic에 설정
  .translate([svgWidth/2, svgHeight/2])	// 화면에서의 표시 위치 조정
  .clipAngle(90)	// 클립 범위 지정
  .scale(280)	// 척도 지정
  .rotate([degree, -25])	// 회전 각도 지정
var path = d3.geo.path()	// 패스와 투영 방법 설정
  .projection(earth)
// 지구의 데이터를 불러옴
d3.json("data/world.json", function(error, world) {
	var earthPath = d3.select("#myGraph")
	  .selectAll("path")	// path 요소 지정
	  .data(world.features)	// 데이터를 저장
	  .enter()
	  .append("path")	// path를 추가
	  .attr("d", path)	// 지형 데이터 설정
	  .style("fill", function(d, i){
			if (d.properties.name == "Antarctica"){	// 남극일 때의 처리
				return "#fff";
			}
			if (d.properties.name == "Korea"){	// 한국일 때의 처리
				return "red";
			}
			return "#eee";	// 한국 이외는 회색으로
		})
	// 타이머를 사용하여 지구를 회전시킴
	d3.timer(function(){
		earth.rotate([degree, -25]);	// 각도 설정
		degree = degree + 2;	// 2도씩 움직임
		earthPath.attr("d", path)	// 지형 데이터 설정
	});
})
