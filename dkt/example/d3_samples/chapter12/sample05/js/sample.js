(function(){
	// SVG 요소의 넓이과 높이를 구함
	var svgEle = document.getElementById("myGraph");
	var svgWidth = window.getComputedStyle(svgEle, null).getPropertyValue("width");
	var svgHeight = window.getComputedStyle(svgEle, null).getPropertyValue("height");
	svgWidth = parseFloat(svgWidth);	// 값에 단위가 포함되었으므로 단위를 삭제
	svgHeight = parseFloat(svgHeight);	// 값에 단위가 포함되었으므로 단위를 삭제
	var pack;		// 원을 패킹하는 g 요소 일람
	var circles;	// circle 요소 일람
	var texts;	// text 요소 일람
	var bubble;	// 팩 레이아웃 객체	
	var allData;	// 모든 데이터를 저장할 변수
	var year = "year2000";	// 표시할 데이터를 나타내는 변수. 최초에는 2005년 데이터
	// JSON 데이터를 불러와 처리
	d3.json("data.json", function(error, data){
		allData = data;
		drawPackLayout(data);
	});
	// 팩 레이아웃 표시
	function drawPackLayout(dataSet){
		// 색을 준비
		var color = d3.scale.category10();  // 10색을 지정
		// 팩 레이아웃
		bubble = d3.layout.pack()
		  .size([320, 320])	// 표시 크기 지정
		// 팩 레이아웃으로 사용할 크룹을 작성
		pack = d3.select("#myGraph")
		  .selectAll("g")
		  .data(bubble.value(function(d, i){
				return d[year];	
			}).nodes(dataSet))	// 데이터셋을 요소에 설정
		  .enter()
		  .append("g")
		// 원 생성
		circles = pack.append("circle")
		  .attr("r", 0)	// 최초 반지름은 0으로 함
		  .attr("transform", function(d, i){
				return "translate(" + d.x + "," + d.y + ")";	// X,Y 좌표 설정
			})
		  .style("fill", function(d, i){
				return color(i);
			})
		circles.transition()
		  .duration(function(d, i){	// 자식의 깊이에 따라 대기 시간을 설정
				return d.depth * 1000 + 500;	// 깊이×1초+0.5초
			})
		  .attr("r", function(d){	// 반지름 지정
				return d.r;
			})
		// 원에 표시할 문자 생성
		texts = pack.append("text")
		  .attr("transform", function(d, i){
				return "translate(" + d.x + "," + d.y + ")";	// X,Y 좌표 설정
			})
		texts.style("opacity", 1.0)	// 투명으로 함
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
	// 버튼 클릭으로 데이터를 불러와 애니메이션 효과와 함께 표시
	d3.selectAll("input").on("click", function(){
		year = d3.select(this).attr("data-year");
		pack.data(bubble.value(function(d, i){
			return d[year];	
		}).nodes(allData))	// 데이터셋을 요소에 설정
		circles
		  .transition()	// 속성을 애니메이션을 사용하여 변경
		  .duration(500)	// 0.5초간 애니메이션
		  .ease("bounce")	// 통통 튀는 움직임으로 함
		  .attr("r", function(d, i){	// 원의 반지름을 설정
				return d.r;
			})
		  .attr("transform", function(d, i){
				return "translate(" + d.x + "," + d.y + ")";	// X,Y 좌표 설정
			})
		texts
		  .transition()	// 속성을 애니메이션을 사용하여 변경
		  .duration(500)	// 0.5초간 애니메이션
		  .ease("bounce")
		  .attr("transform", function(d, i){
				return "translate(" + d.x + "," + d.y + ")";	// X,Y 좌표 설정
			})
	});
})();
