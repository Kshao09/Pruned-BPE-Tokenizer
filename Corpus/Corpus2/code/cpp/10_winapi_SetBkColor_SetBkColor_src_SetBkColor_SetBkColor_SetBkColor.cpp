// wb_t@CÌCN[h
#include <windows.h>	// WWindowsAPI
#include <tchar.h>		// TCHAR^
#include <string.h>		// C¶ñ

// ÖÌvg^Cvé¾
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);	// EBhEbZ[WÉÎµÄÆ©Ìð·éæ¤Éè`µ½R[obNÖWindowProc.

// _tWinMainÖÌè`
int WINAPI _tWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR lpCmdLine, int nShowCmd) {

	// ÏÌé¾
	HWND hWnd;			// CreateWindowÅì¬µ½EBhEÌEBhEnhði[·éHWND^ÏhWnd.
	MSG msg;			// EBhEbZ[Wîñði[·éMSG\¢Ì^Ïmsg.
	WNDCLASS wc;		// EBhENXîñðàÂWNDCLASS\¢Ì^Ïwc.

	// EBhENXÌÝè
	wc.lpszClassName = _T("SetBkColor");					// EBhENX¼Í"SetBkColor".
	wc.style = CS_HREDRAW | CS_VREDRAW;						// X^CÍCS_HREDRAW | CS_VREDRAW.
	wc.lpfnWndProc = WindowProc;							// EBhEvV[WÍÆ©Ìðè`µ½WindowProc.
	wc.hInstance = hInstance;								// CX^XnhÍ_tWinMainÌø.
	wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);				// ACRÍAvP[VùèÌàÌ.
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);				// J[\Íîó.
	wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);	// wiÍuV.
	wc.lpszMenuName = NULL;									// j[ÍÈµ.
	wc.cbClsExtra = 0;										// 0Å¢¢.
	wc.cbWndExtra = 0;										// 0Å¢¢.

	// EBhENXÌo^
	if (!RegisterClass(&wc)) {	// RegisterClassÅEBhENXðo^µ, 0ªÔÁ½çG[.

		// G[
		MessageBox(NULL, _T("RegisterClass failed!"), _T("SetBkColor"), MB_OK | MB_ICONHAND);	// MessageBoxÅ"RegisterClass failed!"ÆG[bZ[Wð\¦.
		return -1;	// ÙíI¹(1)

	}

	// EBhEÌì¬
	hWnd = CreateWindow(_T("SetBkColor"), _T("SetBkColor"), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, NULL, NULL, hInstance, NULL);	// CreateWindowÅ, ãÅo^µ½"SetBkColor"EBhENXÌEBhEðì¬.
	if (hWnd == NULL) {	// EBhEÌì¬É¸sµ½Æ«.

		// G[
		MessageBox(NULL, _T("CreateWindow failed!"), _T("SetBkColor"), MB_OK | MB_ICONHAND);	// MessageBoxÅ"CreateWindow failed!"ÆG[bZ[Wð\¦.
		return -2;	// ÙíI¹(2)

	}

	// EBhEÌ\¦
	ShowWindow(hWnd, SW_SHOW);	// ShowWindowÅSW_SHOWðwèµÄEBhEÌ\¦.

	// bZ[W[v
	while (GetMessage(&msg, NULL, 0, 0) > 0) {	// GetMessageÅbZ[Wðæ¾, ßèlª0æèå«¢ÔÍ[vµ±¯é.

		// EBhEbZ[WÌo
		DispatchMessage(&msg);	// DispatchMessageÅó¯æÁ½bZ[WðEBhEvV[W(±ÌêÍÆ©Éè`µ½WindowProc)Éo.

	}

	// vOÌI¹
	return (int)msg.wParam;	// I¹R[h(msg.wParam)ðßèlÆµÄÔ·.

}

// WindowProcÖÌè`
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {	// EBhEbZ[WÉÎµÄÆ©Ìð·éæ¤Éè`µ½EBhEvV[W.

	// EBhEbZ[WÉÎ·é.
	switch (uMsg) {	// switch-casa¶ÅuMsgÌl²ÆÉðUèª¯é.

		// EBhEÌì¬ªJn³ê½.
		case WM_CREATE:		// EBhEÌì¬ªJn³ê½.(uMsgªWM_CREATEÌ.)

			// WM_CREATEubN
			{

				// EBhEì¬¬÷
				return 0;	// return¶Å0ðÔµÄ, EBhEì¬¬÷Æ·é.

			}

			// ùèÌÖü©¤.
			break;	// breakÅ²¯Ä, ùèÌ(DefWindowProc)Öü©¤.

		// EBhEªjü³ê½.
		case WM_DESTROY:	// EBhEªjü³ê½.(uMsgªWM_DESTROYÌ.)

			// WM_DESTROYubN
			{

				// I¹bZ[WÌM.
				PostQuitMessage(0);	// PostQuitMessageÅI¹R[hð0ÆµÄWM_QUITbZ[WðM.(·éÆbZ[W[vÌGetMessageÌßèlª0ÉÈéÌÅ, bZ[W[v©ç²¯é.)

			}

			// ùèÌÖü©¤.
			break;	// breakÅ²¯Ä, ùèÌ(DefWindowProc)Öü©¤.

		// æÊÌ`æðv³ê½.
		case WM_PAINT:		// æÊÌ`æðv³ê½.(uMsgªWM_PAINTÌ.)

			// WM_PAINTubN
			{

				// ±ÌubNÌ[JÏEzñÌé¾Æú».
				HDC hDC;		// foCXReLXgnhði[·éHDC^ÏhDC.
				PAINTSTRUCT ps;	// yCgîñðÇ·éPAINTSTRUCT\¢Ì^ÌÏps.
				TCHAR tszText[] = _T("ABCDE");	// TCHAR^zñtszTextð"ABCDE"Åú».
				size_t uiLen = 0;	// tszTextÌ·³ði[·é½ßÌsize_t^ÏuiLenð0Éú».

				// EBhEÌ`æJn
				hDC = BeginPaint(hwnd, &ps);	// BeginPaintÅ±ÌEBhEÌ`æÌõð·é. ßèlÉÍfoCXReLXgnhªÔéÌÅ, hDCÉi[.

				// wiFÌÝè
				SetBkColor(hDC, RGB(0x00, 0x00, 0xff));	// SetBkColorÅÂðZbg.

				// `æFÌÝè
				SetTextColor(hDC, RGB(0xff, 0x00, 0x00));	// SetTextColorÅÔðZbg.

				// ¶ñÌ`æ
				uiLen = _tcslen(tszText);	// _tcslenÅtszTextÌ·³ðæ¾µ, uiLenÉi[.
				TextOut(hDC, 50, 50, tszText, (int)uiLen);	// TextOutÅEBhEhwndÌÀW(50, 50)ÌÊuÉtszTextð`æ.

				// EBhEÌ`æI¹
				EndPaint(hwnd, &ps);	// EndPaintÅ±ÌEBhEÌ`æðI¹·é.

			}

			// ùèÌÖü©¤.
			break;	// breakÅ²¯Ä, ùèÌ(DefWindowProc)Öü©¤.

		// ãLÈOÌ.
		default:	// ãLÈOÌlÌÌùè.

			// ùèÌÖü©¤.
			break;	// breakÅ²¯Ä, ùèÌ(DefWindowProc)Öü©¤.

	}

	//  ÆÍùèÌÉC¹é.
	return DefWindowProc(hwnd, uMsg, wParam, lParam);	// ßèlàÜßDefWindowProcÉùèÌðC¹é.

}