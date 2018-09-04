#pragma once
template<typename T> struct token;

template<typename T>
struct token{
	token<T> *prev;
	T item;
	token<T> *next;
};

template<typename T>
class Ptr_List {
private:
	uint32_t length;

	token<T> *start;
	token<T> *curr;
	token<T> *end;
public:
	Ptr_List();
	~Ptr_List();
	uint32_t size() { return length; }
	void push_back(T item);
	void insert(T item, uint32_t loc);
	void remove(int index);
	void clear();
	T* iterator();
	T* next();
	T operator[](int ind);
};

template<typename T>
inline Ptr_List<T>::Ptr_List(){
	start=(token<T>*)malloc(sizeof(token<T>));
	start->prev = NULL;
	start->item = NULL;
	start->next = NULL;

	length = 0;

	curr=end=start;
}

template<typename T>
inline Ptr_List<T>::~Ptr_List<T*>() {
	curr = start;
	while (curr != end) {
		token<T> *del = curr;
		curr = curr->next;
		free(del->item);
		free(del);
	}
	if (end->item != NULL) {
		free(end->item);
	}
	free(end);
}

template<typename T>
inline void Ptr_List<T>::insert(T item, uint32_t loc){
	curr = start;
	for (int i = 0; i < loc; i++) {
		curr = curr->next;
	}
	token<T> *temp = (token<T>)malloc(sizeof(token<T>));
	if (curr->prev) {
		temp->prev = curr->prev;
		curr->prev->next = temp;
	}else {
		temp->prev = NULL;
		start = temp;
	}

	if (curr) {
		temp->next = curr;
		curr->prev = temp;
	}else {
		temp->next = NULL;
		end = temp;
	}
	length++;
}

template<typename T>
inline void Ptr_List<T>::remove(int index){
	curr = start;
	for (int i = 0; i < index; i++) {
		curr = curr->next;
	}
	if (curr->next == NULL && curr->prev == NULL) {
		free(curr->item);
		size--;
		return;
	}
	if (curr->prev == NULL) {
		curr->next->prev = NULL;
		start = curr->next;
	}else {
		curr->prev->next = curr->next;
	}
	if (curr->next = NULL) {
		curr->prev->next = NULL;
		end = curr->prev;
	}else {
		curr->next->prev = curr->prev;
	}
	free(curr->item);
	free(curr);
	size--;
}

template<typename T>
inline void Ptr_List<T>::clear(){
	curr = start->next;
	while (curr != end) {
		token<T> *del = curr;
		curr = curr->next;
		free(del->item);
		free(del);
	}
	if (end->item != NULL) {
		free(end->item);
	}
	free(end);
	free(start->item);
	curr = start;
	end = start;
	length = 0;
}

template<typename T>
inline T * Ptr_List<T>::iterator(){
	curr = start;
	return &curr->item;
}

template<typename T>
inline T* Ptr_List<T>::next(){
	if (curr != NULL) {
		curr = curr->next;
	}
	return  &curr->item;
}

template<typename T>
inline T Ptr_List<T>::operator[](int ind){
	curr = start;
	for (int i = 0; i < ind; i++) {
		curr = curr->next;
	}
	return curr;
}

template<typename T>
inline void Ptr_List<T>::push_back(T item){
	if (length == 0) {
		start->item = item;
	}else {
		end->item = item;
		end->next = (token<T>*)malloc(sizeof(token<T>));
		end->next->prev = end;
		end = end->next;
		end->next = NULL;
	}
	length++;
}
