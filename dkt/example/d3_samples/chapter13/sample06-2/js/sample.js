var svgWidth = 800;	// SVG요소의 넓이
var svgHeight = 800;	// SVG요소의 높이
// 조사할 텍스트 파일 불러오기
d3.text("sample.txt", function(error, plainText){
	var count = [ ];	// 문자 종류를 넣을 변수(해시로 사용)
	for(var i=0; i<plainText.length; i++){	// 전체 문자 수만큼 반복
		var c = plainText.charAt(i);	// 1 문자 읽어오기
		if(!count[c]){	// 이미 등록된 문자인가 조사
			count[c] = 1;	// 처음 문자가 나옴
		}else{
			count[c] = count[c] + 1;	// 카운트를 늘임
		}
	}
	var temp = [ ];	// 일시적인 배열 변수
	for(i in count){	// 문자의 종류만큼 반복
		temp.push({ name : i, value : count[i] });	// 문자 이름과 표시 수를 입력
	}
	// 데이터셋 생성
	var dataSet = {
		children: temp
	};
	drawTreemap(dataSet);	// 트리맵 표시
})
// 트리맵 표시할 함수
function drawTreemap(dataSet){
	// 트리맵 레이아웃
	var treemap = d3.layout.treemap()
	  .size([svgWidth, svgHeight])   // SVG 요소의 넓이에 맞춤
	  .mode("slice")
	// 트리맵 그리기
	var tmap = d3.select("#myGraph")
	  .selectAll("rect")   // div에 표시할 박스 할당
	  .data(treemap.nodes(dataSet))    // 노드를 대상으로 처리
	// 분할 맵 영역 추가
	tmap.enter()
	  .append("rect")  // rect 요소를 추가
	  .attr("class", "block")	// CSS 클래스 추가
	  .attr("x", function(d, i) { // X 좌표 설정
			return d.x;
		}) 
	  .attr("y", function(d, i) { // Y 좌표 설정
			return d.y;
		})
	  .attr("width", function(d, i) {	// 넓이 설정
			return d.dx;
		})
	  .attr("height", function(d, i) {	// 높이 설정
			return d.dy;
		})
	// 맵 안에 문자 추가
	tmap.enter()
	  .append("text")	// text요소를 추가
	  .attr("class", "name")	// CSS 클래스 추가
	  .attr("transform", function(d, i){	// 위치를 계산하여 XY 좌표를 일괄 설정
			return "translate(" + (d.x+d.dx/2) + "," + (d.y+d.dy/2) + ")";	// X,Y 좌표 설정
		})
	  .attr("dy", "0.2em")	// 표시 위치 조정
	  .text(function(d, i) {  // 문자 표시
			return d.name;	// 영역 안에 표시할 문자를 반환
		})
	  .style("font-size", function(d, i){	// 영역의 크기에 따라 문자 크기를 조정
			return (d.dx * d.dy) / 200;
		})
}
