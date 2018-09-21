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
	uint32_t count;

	unique_token<T> *start;
	unique_token<T> *curr;
	unique_token<T> *end;
public:
	Ptr_Set();
	~Ptr_Set();

	bool insert(T item);
	void remove(T item,bool del);
	T* iterator();
	T* next();
	T head() { return start->item; }
	uint32_t size() { return count; }
	void clear(bool free_ptrs);
};

template<typename T>
inline Ptr_Set<T>::Ptr_Set(){
	start = (unique_token<T>*)malloc(sizeof(unique_token<T>));
	end = start;
	curr = start;

	start->prev = NULL;
	start->next = NULL;

	count = 0;
}

template<typename T>
inline Ptr_Set<T>::~Ptr_Set(){
	clear(true);
	free(start);
}

template<typename T>
inline bool Ptr_Set<T>::insert(T item){
	if (count == 0) {
		start->item = item;
		count++;
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

		count++;
		return true;
	}
}

template<typename T>
inline void Ptr_Set<T>::remove(T item, bool del){
	curr = start;
	while (curr != NULL) {
		if (curr->item == item) {
			if (del) {
				free(curr->item);
			}

			if (count == 1) {
				count--;
				curr->item = NULL;
				return;
			}
			//If we found the item at the end.
			if (!curr->next) {
				curr->prev->next = NULL;
				curr = end;
			}else {
				curr->prev->next = curr->next;
			}
			//If we found the item at the start.
			if (!curr->prev) {
				curr->next->prev = NULL;
				curr = start;
			}else {
				curr->next->prev = curr->prev;
			}

			count--;
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
	return &curr->item;
}

template<typename T>
inline void Ptr_Set<T>::clear(bool free_ptrs){
	curr = start;
	count = 0;
	while (curr != NULL) {
		if (curr->next = NULL) {
			if (free_ptrs) {
				free(curr->item);
			}
			break;
		}else {
			if (free_ptrs) {
				free(curr->item);
			}
			unique_token<T> *del = curr;
			curr = curr->next;
			free(del);
		}
	}
	start = curr;
	end = curr;
}
