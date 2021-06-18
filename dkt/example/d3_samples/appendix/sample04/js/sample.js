// 계두도의 데이터셋
var dataSet = [
	{ location: "교정", value : 90 },
	{ location: "자택", value : 35 },
	{ location: "교사", value : 70 },
	{ location: "논밭 등", value : 25 },
	{ location: "기타", value : 60 }
];
// 그래프 관계의 데이터를 변수에 설정
var svgWidth = 320;	// svg 요소의 높이
var svgHeight = 320;	// svg 요소의 높이
var iRadius = 50;	// 안쪽 원의 반지름
var oRadius = iRadius + 10;	// 안쪽 원 + 오프셋의 반지름
var color = d3.scale.category10();	// D3.js의 색 사용
// 계두도 하나의 부채꼴의 비율로 각도를 계산(360도를 데이터 수로 나눔)
for(var i=0; i<dataSet.length; i++){
	dataSet[i].startAngle = (360/dataSet.length)*i * Math.PI / 180;
	dataSet[i].endAngle = (360/dataSet.length)*(i+1) * Math.PI / 180;
}
// 계두도 원 크기 지정
var arc = d3.svg.arc()
  .innerRadius(iRadius)
  .outerRadius(function(d){ return oRadius + d.value; })	// 반지름을 데이터마다 지정
// 계두도 그리기
d3.select("#myGraph")
  .selectAll("path")
  .data(dataSet)	// 데이터를 세팅
  .enter()
  .append("path") // 원호는 패스로 지정
  .attr("class", "pie")	// CSS 클래스에 pie를 지정
  .attr("d", arc)	// 원호 설정
  .attr("fill", function(d, i){
		return color(i);	// 칠하기 설정
	})
  .attr("transform", "translate("+svgWidth/2+", "+svgHeight/2+")")	// 그래프 중심을 표시
// 문자 그리기
d3.select("#myGraph")
  .selectAll("text")
  .data(dataSet)	// 데이터 세팅
  .enter()
  .append("text") // 원호는 패스로 지정
  .attr("class", "label")	// CSS 클래스에 pie를 지정
  .attr("transform", function(d, i){	// 표시 위치 지정
		var c = arc.centroid(d);	// 부채꼴 중심을 구함
		var x = c[0] + svgWidth/2;	// X 좌표 읽어오기
		var y = c[1] + svgHeight/2;	// Y 좌표 읽어오기
		return "translate("+x+", "+y+")";
	})
  .text(function(d, i){	// 문자 표시
		return d.location + "(" + d.value + ")";
	})
