class HDRImage {
public:
	int mSizeX, mSizeY;
	void setSize(int, int);
	void loadPfm(const char *);
	void writePfm(const char *);
	float * data;
};