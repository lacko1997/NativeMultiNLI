#pragma once
#pragma once

template<typename T>
class List {
private:
	uint32_t length;

	token<T> *start;
	token<T> *curr;
	token<T> *end;
public:
	List();
	~List();
	uint32_t size() { return length; }
	void push_back(T item);
	void insert(T item, uint32_t loc);
	void remove(int index);
};

template<typename T>
inline List<T>::List() {
	start = (token<T>*)malloc(sizeof(token<T>));
	start->prev = NULL;
	start->next = NULL;

	length = 0;

	curr = end = start;
}

template<typename T>
inline List<T>::~List() {
	curr = start;
	while (curr != end) {
		token<T> *del = curr;
		curr = curr->next;
		free(del);
	}
	free(end);
}

template<typename T>
inline void List<T>::insert(T item, uint32_t loc) {
	for (int i = 0; i < loc; i++) {
		curr = curr->next;
	}
	token<T> *temp = (token<T>)malloc(sizeof(token<T>));
	if (curr->prev) {
		temp->prev = curr->prev;
		curr->prev->next = temp;
	}
	else {
		temp->prev = NULL;
		start = temp;
	}

	if (curr) {
		temp->next = curr;
		curr->prev = temp;
	}
	else {
		temp->next = NULL;
		end = temp;
	}
	curr = start;
	length++;
}

template<typename T>
inline void List<T>::remove(int index) {
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
inline void List<T>::push_back(T item) {
	if (length == 0) {
		start->item = item;
	}
	else {
		end->item = item;
		end->next = (token<T>*)malloc(sizeof(token<T>));
		end->next->prev = end;
		end = end->next;
		end->next = NULL;
	}
	length++;
}
