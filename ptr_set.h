#pragma once

template<typename T>
struct unique_token;

template<typename T>
struct unique_token {
	unique_token<T>* prev;
	T item;
	unique_token<T>* next;
};

template <typename T>
class Ptr_Set{
private:
	uint32_t size;

	unique_token<T> *start;
	unique_token<T> *curr;
	unique_token<T> *end;
public:
	Ptr_Set();
	~Ptr_Set();
	bool insert(T item);
	void remove(T item);
	T* iterator();
	T* next();
	void clear();
};

template<typename T>
inline Ptr_Set<T>::Ptr_Set(){
	start = (unique_token<T>*)malloc(sizeof(unique_token<T>));
	end = start;
	curr = start;

	start->prev = NULL;
	start->next = NULL;

	size = 0;
}

template<typename T>
inline Ptr_Set<T>::~Ptr_Set(){
	clear();
	free(start);
}

template<typename T>
inline bool Ptr_Set<T>::insert(T item){
	if (size == 0) {
		start->item = item;
		return true;
	}else {
		curr = start;
		while (curr != NULL) {
			if (curr->item == item) {
				return false;
			}
			curr = curr->next;
		}
		curr = (unique_token<T>*)malloc(sizeof(unique_token<T>));

		curr->item = NULL;
		curr->prev = end;
		curr->prev->next = curr;
		curr->next = NULL;

		end = curr;
		return true;
	}
}

template<typename T>
inline void Ptr_Set<T>::remove(T item){
	curr = start;
	while (curr != NULL) {
		if (curr->item == item) {
			free(curr->item);
			if (size == 1) {
				size--;
				return;
			}
			if (!curr->next) {
				curr->prev->next = NULL;
				curr = end;
			}else {
				curr->prev->next = curr->next;
			}
			if (!curr->prev) {
				curr->next->prev = NULL;
				curr = start;
			}else {
				curr->next->prev = curr->prev;
			}
			size--;
			free(curr);
			return;
		}
		curr = curr->next;
	}
}

template<typename T>
inline T * Ptr_Set<T>::iterator(){
	curr = start;
	return &curr->item;
}

template<typename T>
inline T * Ptr_Set<T>::next(){
	if (curr != NULL) {
		curr = curr->next;
	}else {
		return NULL;
	}
	return cur->item;
}

template<typename T>
inline void Ptr_Set<T>::clear(){
	curr = start;
	size = 0;
	while (curr != NULL) {
		if (curr->next = NULL) {
			free(curr->item);
			break;
		}else {
			free(curr->item);
			unique_token<T> *del = curr;
			curr = curr->next;
			free(del);
		}
	}
	start = curr;
	end = curr;
}
