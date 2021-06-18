// 변수 설정
svgWidth = 320;	// svg 요소의 넓이
svgHeight = 240;	// svg 요소의 높이
var dataSet = [20, 20, 20, 20, 20];	// 데이터셋 준비
// 1분마다 데이터를 읽어와 처리
setInterval("drawPie()", 1000*60);
drawPie();	// 처음에만 읽어와 원 그래프를 표시해둠
// 원 그래프 처리
var pie = d3.layout.pie();	// 원 그래프 레이아웃
var arc = d3.svg.arc().outerRadius(100);	// 원 그래프의 바깥 지름 설정
// 원 그래프 그리기
var groupElements = d3.select("#myGraph")
  .selectAll("g")	// g 요소 지정
  .data(pie(dataSet))	// 데이터를 요소에 연결
  .enter()
  .append("g")	// 중심 계산을 위한 그룹 작성
  .attr("transform", "translate("+svgWidth/2+", "+svgHeight/2+")")    // 원 그래프의 중심으로 함
// 데이터 추가
var pieElements = groupElements	// 부채꼴 부분
  .append("path")	// 데이터 수만큼 path 요소가 추가됨
  .attr("class", "pie")	// CSS 클래스를 지정
  .attr("d", arc)	// 부채꼴 지정
  .style("fill", function(d, i){
		return ["#55f", "red", "black", "green", "#ddd"][i];	// 대응하는 OS의 색을 반환
	})
// 데이터를 불러와 원 그래프 표시
function drawPie(){
	// 데이터셋은 단순 텍스트 파일
	d3.text("log.txt", function(error, plainText){
		// 문자열을 검색하여 간단히 점유율을 구함
		var win = (plainText.match(/Windows/g) || "").length;
		var mac = (plainText.match(/Macintosh/g) || "").length;
		var iphone = (plainText.match(/ iPhone/g) || "").length;
		var and = (plainText.match(/ Android/g) || "").length;
		var etc = 100 - win - mac - iphone - and;	// 기타
		dataSet = [win, mac, iphone, and, etc];	// 데이터셋 준비
		dataSet = [30, 20, 10, 30, 20];	// 데이터셋 준비
		d3.select("#time").text(new Date());	// 날짜 표시
		// 데이터 업데이트
		pieElements
		  .data(pie(dataSet))	// 데이터를 요소에 연결
		  .attr("d", arc)	// 부채꼴 표시
	})
}
